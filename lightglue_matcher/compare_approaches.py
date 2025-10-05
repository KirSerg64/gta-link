"""
Quick Comparison: Original vs LightGlue Approach

This script helps you quickly compare results from both approaches.

Usage:
    python compare_approaches.py \
        --original_dir ./results_original \
        --lightglue_dir ./results_lightglue
"""

import argparse
import os
from collections import defaultdict
from loguru import logger


def parse_mot_file(file_path):
    """Parse MOT format tracking file"""
    data = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bbox = [float(x) for x in parts[2:6]]
            
            data[frame_id].append({
                'track_id': track_id,
                'bbox': bbox
            })
    
    return data


def count_tracklets(data):
    """Count unique tracklets"""
    track_ids = set()
    for frame_detections in data.values():
        for det in frame_detections:
            track_ids.add(det['track_id'])
    return len(track_ids)


def count_id_switches(data):
    """Count potential ID switches (bbox overlap between tracks)"""
    switches = 0
    for frame_id, detections in data.items():
        # Check for overlapping bboxes with different IDs
        for i, det1 in enumerate(detections):
            for det2 in detections[i+1:]:
                if det1['track_id'] != det2['track_id']:
                    # Check IoU
                    iou = compute_iou(det1['bbox'], det2['bbox'])
                    if iou > 0.5:  # Significant overlap
                        switches += 1
    return switches


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to x1, y1, x2, y2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1+w1, y1+h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2+w2, y2+h2
    
    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def compute_tracklet_lengths(data):
    """Compute average tracklet length"""
    tracklet_frames = defaultdict(set)
    
    for frame_id, detections in data.items():
        for det in detections:
            tracklet_frames[det['track_id']].add(frame_id)
    
    lengths = [len(frames) for frames in tracklet_frames.values()]
    
    if not lengths:
        return 0, 0, 0
    
    return min(lengths), sum(lengths)/len(lengths), max(lengths)


def compare_files(original_file, lightglue_file):
    """Compare two result files"""
    logger.info(f"\nComparing: {os.path.basename(original_file)}")
    
    # Parse files
    original_data = parse_mot_file(original_file)
    lightglue_data = parse_mot_file(lightglue_file)
    
    # Compute statistics
    orig_num_tracklets = count_tracklets(original_data)
    lg_num_tracklets = count_tracklets(lightglue_data)
    
    orig_min, orig_avg, orig_max = compute_tracklet_lengths(original_data)
    lg_min, lg_avg, lg_max = compute_tracklet_lengths(lightglue_data)
    
    # Print comparison
    logger.info("=" * 60)
    logger.info(f"{'Metric':<30} {'Original':<15} {'LightGlue':<15}")
    logger.info("=" * 60)
    logger.info(f"{'Number of tracklets':<30} {orig_num_tracklets:<15} {lg_num_tracklets:<15}")
    logger.info(f"{'Avg tracklet length':<30} {orig_avg:<15.1f} {lg_avg:<15.1f}")
    logger.info(f"{'Min tracklet length':<30} {orig_min:<15} {lg_min:<15}")
    logger.info(f"{'Max tracklet length':<30} {orig_max:<15} {lg_max:<15}")
    
    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("Interpretation:")
    logger.info("=" * 60)
    
    if lg_num_tracklets < orig_num_tracklets:
        reduction = (orig_num_tracklets - lg_num_tracklets) / orig_num_tracklets * 100
        logger.info(f"✅ LightGlue reduced fragmentation by {reduction:.1f}%")
    elif lg_num_tracklets > orig_num_tracklets:
        increase = (lg_num_tracklets - orig_num_tracklets) / orig_num_tracklets * 100
        logger.info(f"⚠️  LightGlue increased tracklets by {increase:.1f}% (may need tuning)")
    else:
        logger.info("ℹ️  Same number of tracklets")
    
    if lg_avg > orig_avg:
        increase = (lg_avg - orig_avg) / orig_avg * 100
        logger.info(f"✅ LightGlue increased avg length by {increase:.1f}%")
    elif lg_avg < orig_avg:
        decrease = (orig_avg - lg_avg) / orig_avg * 100
        logger.info(f"⚠️  LightGlue decreased avg length by {decrease:.1f}%")
    
    return {
        'original_tracklets': orig_num_tracklets,
        'lightglue_tracklets': lg_num_tracklets,
        'original_avg_length': orig_avg,
        'lightglue_avg_length': lg_avg
    }


def main():
    parser = argparse.ArgumentParser(description="Compare original vs LightGlue results")
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory with original results')
    parser.add_argument('--lightglue_dir', type=str, required=True,
                       help='Directory with LightGlue results')
    args = parser.parse_args()
    
    # Check directories exist
    if not os.path.exists(args.original_dir):
        logger.error(f"Original directory not found: {args.original_dir}")
        return
    
    if not os.path.exists(args.lightglue_dir):
        logger.error(f"LightGlue directory not found: {args.lightglue_dir}")
        return
    
    # Get list of files
    original_files = sorted([f for f in os.listdir(args.original_dir) if f.endswith('.txt')])
    lightglue_files = sorted([f for f in os.listdir(args.lightglue_dir) if f.endswith('.txt')])
    
    # Find common files
    common_files = set(original_files) & set(lightglue_files)
    
    if not common_files:
        logger.error("No common files found between directories")
        return
    
    logger.info("=" * 60)
    logger.info("Results Comparison: Original vs LightGlue")
    logger.info("=" * 60)
    logger.info(f"Original dir: {args.original_dir}")
    logger.info(f"LightGlue dir: {args.lightglue_dir}")
    logger.info(f"Common files: {len(common_files)}")
    
    # Compare each file
    all_results = []
    for filename in sorted(common_files):
        original_file = os.path.join(args.original_dir, filename)
        lightglue_file = os.path.join(args.lightglue_dir, filename)
        
        results = compare_files(original_file, lightglue_file)
        all_results.append(results)
    
    # Overall summary
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 60)
        
        avg_orig_tracklets = sum(r['original_tracklets'] for r in all_results) / len(all_results)
        avg_lg_tracklets = sum(r['lightglue_tracklets'] for r in all_results) / len(all_results)
        avg_orig_length = sum(r['original_avg_length'] for r in all_results) / len(all_results)
        avg_lg_length = sum(r['lightglue_avg_length'] for r in all_results) / len(all_results)
        
        logger.info(f"Average tracklets - Original: {avg_orig_tracklets:.1f}, LightGlue: {avg_lg_tracklets:.1f}")
        logger.info(f"Average length - Original: {avg_orig_length:.1f}, LightGlue: {avg_lg_length:.1f}")
        
        if avg_lg_tracklets < avg_orig_tracklets:
            reduction = (avg_orig_tracklets - avg_lg_tracklets) / avg_orig_tracklets * 100
            logger.info(f"\n✅ Overall: LightGlue reduced fragmentation by {reduction:.1f}%")
        
        if avg_lg_length > avg_orig_length:
            increase = (avg_lg_length - avg_orig_length) / avg_orig_length * 100
            logger.info(f"✅ Overall: LightGlue increased tracklet length by {increase:.1f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("Recommendation:")
    logger.info("=" * 60)
    logger.info("If LightGlue shows improvement:")
    logger.info("  → Use LightGlue for production")
    logger.info("If results are similar:")
    logger.info("  → Consider tuning --merge_dist_thres")
    logger.info("  → Try --lightglue_sample_strategy adaptive")
    logger.info("  → Increase --lightglue_samples")
    logger.info("If LightGlue is worse:")
    logger.info("  → Check video quality")
    logger.info("  → Enable --use_clahe")
    logger.info("  → Review log messages for errors")


if __name__ == "__main__":
    main()
