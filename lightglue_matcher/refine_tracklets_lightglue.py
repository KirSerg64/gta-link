"""
Global Tracklet Association with LightGlue-based Distance Calculation

This is a modified version of refine_tracklets.py that uses image-based 
keypoint matching (SuperPoint + LightGlue) instead of cosine distance on 
pre-computed ReID features.

Key Differences from Original:
- Uses LightGlue for tracklet distance calculation
- Requires video file path (not just pkl files)
- More robust to domain shift and appearance variations
- Optimized for A100 GPU with batching and caching

Usage:
    python refine_tracklets_lightglue.py \
        --dataset SoccerNet \
        --tracker SORT \
        --track_src ./tracklets \
        --video_path ./video.mp4 \
        --use_connect \
        --merge_dist_thres 0.5 \
        --lightglue_max_keypoints 2048 \
        --lightglue_samples 10

Author: Modified from original refine_tracklets.py
Date: 2025-10-05
"""

import numpy as np
import os
import torch
import pickle

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tracklet import Tracklet
from lightglue_matcher import TrackletLightGlueMatcher, get_distance_lightglue

import argparse


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
    """
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments


def query_subtracks(seg1, seg2, track1, track2):
    """
    Processes and pairs up segments from two different tracks to form valid subtracks based on their temporal alignment.

    Args:
        seg1 (list of tuples): List of segments from the first track where each segment is a tuple of start and end indices.
        seg2 (list of tuples): List of segments from the second track similar to seg1.
        track1 (Tracklet): First track object containing times and bounding boxes.
        track2 (Tracklet): Second track object similar to track1.

    Returns:
        list: Returns a list of subtracks which are either segments of track1 or track2 sorted by time.
    """
    subtracks = []
    while seg1 and seg2:
        s1_start, s1_end = seg1[0]
        s2_start, s2_end = seg2[0]

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]
        s2_startFrame = track2.times[s2_start]

        if s1_startFrame < s2_startFrame:
            assert track1.times[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            assert s1_startFrame >= track2.times[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)
    
    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if(s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks


def get_subtrack(track, s_start, s_end):
    """
    Extracts a subtrack from a given track.

    Args:
    track (STrack): The original track object from which the subtrack is to be extracted.
    s_start (int): The starting index of the subtrack.
    s_end (int): The ending index of the subtrack.

    Returns:
    STrack: A subtrack object extracted from the original track object, containing the specified time intervals
            and bounding boxes. The parent track ID is also assigned to the subtrack.
    """
    subtrack = Tracklet()
    subtrack.times = track.times[s_start : s_end + 1]
    subtrack.bboxes = track.bboxes[s_start : s_end + 1]
    subtrack.parent_id = track.track_id

    return subtrack


def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes across all tracks.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        factor (float): Factor by which to scale the calculated x and y ranges.

    Returns:
        tuple: Maximal x and y range scaled by the given factor.
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            assert len(bbox) == 4
            x, y, w, h = bbox[0:4]
            x += w / 2
            y += h / 2
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def display_Dist(Dist, seq_name=None, isMerged=False, isSplit=False):
    """
    Displays a heatmap for the distances between tracklets for one or more sequences.

    Args:
        seq2Dist (dict): A dictionary mapping sequence names to their corresponding distance matrices.
        seq_name (str, optional): Specific sequence name to display the heatmap for. If None, displays for all sequences.
        isMerged (bool): Flag indicating whether the distances are post-merge.
        isSplit (bool): Flag indicating whether the distances are post-split.
    """
    split_info = " After Split" if isSplit else " Before Split"
    merge_info = " After Merge" if isMerged else " Before Merge"
    info = split_info + merge_info
    
    plt.figure(figsize=(10, 8))

    sns.heatmap(Dist, cmap='Blues')

    plt.title(f"{seq_name}{info}")
    plt.show()


def get_distance_matrix_lightglue(tid2track, matcher):
    """
    Constructs and returns a distance matrix between all tracklets using LightGlue keypoint matching.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        matcher (TrackletLightGlueMatcher): Initialized LightGlue matcher

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    logger.info(f"Computing distance matrix using LightGlue for {len(tid2track)} tracklets")
    
    # Use optimized batch computation from matcher
    Dist = matcher.compute_distance_matrix(tid2track)
    
    return Dist


def get_distance(track1_id, track2_id, track1, track2, matcher=None):
    """
    Calculates distance between two tracks.
    
    If matcher is provided, uses LightGlue keypoint matching.
    Otherwise falls back to cosine distance on features.

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.
        matcher (TrackletLightGlueMatcher, optional): LightGlue matcher instance

    Returns:
        float: Distance between the two tracks [0, 1].
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id
    
    # Check for temporal overlap
    doesOverlap = False
    if (track1_id != track2_id):
        doesOverlap = set(track1.times) & set(track2.times)
    
    if doesOverlap:
        return 1.0
    
    # Use LightGlue if matcher provided
    if matcher is not None:
        return matcher.compute_distance(track1, track2)
    
    # Fallback to cosine distance on features
    if not hasattr(track1, 'features') or not hasattr(track2, 'features'):
        logger.warning("No features available and no matcher provided")
        return 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
    track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)
    count1 = len(track1_features_tensor)
    count2 = len(track2_features_tensor)

    cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
    track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
    track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
    cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
    cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator
    
    total_cos_Dist = cos_Dist.sum()
    result = total_cos_Dist / (count1 * count2)
    return result


def check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Checks if two tracklets meet spatial constraints for potential merging.

    Args:
        trk_1 (Tracklet): The first tracklet object containing times and bounding boxes.
        trk_2 (Tracklet): The second tracklet object containing times and bounding boxes, to be evaluated
                        against trk_1 for merging possibility.
        max_x_range (float): The maximum allowed distance in the x-coordinate between the end of trk_1 and
                             the start of trk_2 for them to be considered for merging.
        max_y_range (float): The maximum allowed distance in the y-coordinate under the same conditions as
                             the x-coordinate.

    Returns:
        bool: True if the spatial constraints are met (the tracklets are close enough to consider merging),
              False otherwise.
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)
    
    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    subtrack_1st = subtracks.pop(0)
    
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0 : 4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0 : 4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)
        
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            break
        else:
            subtrack_1st = subtrack_2nd
    
    return inSpatialRange


def merge_tracklets(tracklets, seq2Dist, Dist, seq_name=None, max_x_range=None, max_y_range=None, 
                    merge_dist_thres=None, matcher=None):
    """
    Merge tracklets based on distance matrix using hierarchical clustering.
    
    Args:
        tracklets: Dictionary of tracklets
        seq2Dist: Dictionary to store distance matrices
        Dist: Initial distance matrix
        seq_name: Sequence name
        max_x_range: Maximum spatial x range
        max_y_range: Maximum spatial y range
        merge_dist_thres: Distance threshold for merging
        matcher: LightGlue matcher for recomputing distances after merge
    """
    seq2Dist[seq_name] = Dist

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    
    while (np.any(Dist[non_diagonal_mask] < merge_dist_thres)):
        min_index = np.argmin(Dist[non_diagonal_mask])
        min_value = np.min(Dist[non_diagonal_mask])
        
        masked_indices = np.where(non_diagonal_mask)
        track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]
        
        logger.info(f"Tracks idx to merge: {track1_idx}, {track2_idx} (distance: {min_value:.4f})")

        assert min_value == Dist[track1_idx, track2_idx] == Dist[track2_idx, track1_idx], "Values should match!"

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        
        if inSpatialRange:
            # Merge tracklets
            if hasattr(track1, 'features') and hasattr(track2, 'features'):
                track1.features += track2.features
            track1.times += track2.times
            track1.bboxes += track2.bboxes
            
            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            # Remove merged tracklet from matrix
            Dist = np.delete(Dist, track2_idx, axis=0)
            Dist = np.delete(Dist, track2_idx, axis=1)
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
            
            # Update distance matrix for merged tracklet
            for idx in range(Dist.shape[0]):
                Dist[track1_idx, idx] = get_distance(
                    idx2tid[track1_idx], idx2tid[idx], 
                    tracklets[idx2tid[track1_idx]], tracklets[idx2tid[idx]],
                    matcher=matcher
                )
                Dist[idx, track1_idx] = Dist[track1_idx, idx]
            
            seq2Dist[seq_name] = Dist
            
            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            Dist[track1_idx, track2_idx], Dist[track2_idx, track1_idx] = merge_dist_thres, merge_dist_thres
    
    return tracklets


def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """
    Detects identity switches within a tracklet using clustering.

    Args:
        embs (list of numpy arrays): A list where each element is a numpy array representing an embedding.
                                     Each embedding has the same dimensionality.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        bool: True if an identity switch is detected, otherwise False.
    """
    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)
    
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
    
    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)
            
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(tmp_trklets, eps=None, max_k=None, min_samples=None, len_thres=None):
    """
    Splits each tracklet into multiple tracklets based on an internal distance threshold.

    Args:
        tmp_trklets (dict): Dictionary of tracklets to be processed.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        len_thres (int): Length threshold to filter out short tracklets.
        max_k (int): Maximum number of clusters to consider.

    Returns:
        dict: New dictionary of tracklets after splitting.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres:
            tracklets[tid] = trklet
        else:
            # Only split if features are available
            if hasattr(trklet, 'features') and trklet.features:
                embs = np.stack(trklet.features)
                frames = np.array(trklet.times)
                bboxes = np.stack(trklet.bboxes)
                scores = np.array(trklet.scores)
                
                id_switch_detected, clusters = detect_id_switch(embs, eps=eps, min_samples=min_samples, max_clusters=max_k)
                
                if not id_switch_detected:
                    tracklets[tid] = trklet
                else:
                    unique_labels = set(clusters)

                    for label in unique_labels:
                        if label == -1:
                            continue
                        tmp_embs = embs[clusters == label]
                        tmp_frames = frames[clusters == label]
                        tmp_bboxes = bboxes[clusters == label]
                        tmp_scores = scores[clusters == label]
                        assert new_id not in tmp_trklets
                        
                        tracklets[new_id] = Tracklet(new_id, tmp_frames.tolist(), tmp_scores.tolist(), 
                                                    tmp_bboxes.tolist(), feats=tmp_embs.tolist())
                        new_id += 1
            else:
                # No features - can't split
                tracklets[tid] = trklet

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def save_results(sct_output_path, tracklets):
    """
    Saves the final tracklet results into a specified path.

    Args:
        sct_output_path (str): Path where the results will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.

    """
    results = []

    for i, tid in enumerate(sorted(tracklets.keys())):
        track = tracklets[tid]
        tid = i + 1
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            
            results.append(
                [frame_id, tid, bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1]
            )
    results = sorted(results, key=lambda x: x[0])
    txt_results = []
    for line in results:
        txt_results.append(
            f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]}\n"
            )
    
    with open(sct_output_path, 'w') as f:
        f.writelines(txt_results)
    logger.info(f"save SCT results to {sct_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Global tracklet association with LightGlue-based distance calculation.")
    
    # Original arguments
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset name (e.g., SportsMOT, SoccerNet).')
    
    parser.add_argument('--tracker',
                        type=str,
                        required=True,
                        help='Tracker name.')
    
    parser.add_argument('--track_src',
                        type=str,
                        required=True,
                        help='Source directory of tracklet pkl files.')
    
    parser.add_argument('--use_split',
                        action='store_true',
                        help='If using split component.')
    
    parser.add_argument('--min_len',
                        type=int,
                        default=100,
                        help='Minimum length for a tracklet required for splitting.')
    
    parser.add_argument('--eps',
                        type=float,
                        default=0.7,
                        help='For DBSCAN clustering, the maximum distance between two samples for one to be considered as in the neighborhood of the other.')
    
    parser.add_argument('--min_samples',
                        type=int,
                        default=10,
                        help='The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.')
    
    parser.add_argument('--max_k',
                        type=int,
                        default=3,
                        help='Maximum number of clusters/subtracklets to be output by splitting component.')
    
    parser.add_argument('--use_connect',
                        action='store_true',
                        help='If using connecting component.')
    
    parser.add_argument('--spatial_factor',
                        type=float,
                        default=1,
                        help='Factor to adjust spatial distances.')
    
    parser.add_argument('--merge_dist_thres',
                        type=float,
                        default=0.5,
                        help='Distance threshold for merging tracklets (0-1). Lower = more strict.')
    
    # NEW: LightGlue-specific arguments
    parser.add_argument('--video_path',
                        type=str,
                        required=True,
                        help='Path to video file for extracting frames.')
    
    parser.add_argument('--lightglue_max_keypoints',
                        type=int,
                        default=2048,
                        help='Maximum keypoints for SuperPoint (higher = more accurate but slower).')
    
    parser.add_argument('--lightglue_confidence',
                        type=float,
                        default=0.3,
                        help='Minimum confidence threshold for valid matches.')
    
    parser.add_argument('--lightglue_samples',
                        type=int,
                        default=10,
                        help='Number of frames to sample per tracklet.')
    
    parser.add_argument('--lightglue_sample_strategy',
                        type=str,
                        default='uniform',
                        choices=['uniform', 'adaptive', 'endpoints'],
                        help='Frame sampling strategy: uniform, adaptive (by bbox size), or endpoints.')
    
    parser.add_argument('--lightglue_cache_size',
                        type=float,
                        default=10.0,
                        help='Frame cache size in GB.')
    
    parser.add_argument('--lightglue_batch_size',
                        type=int,
                        default=16,
                        help='Batch size for feature extraction.')
    
    parser.add_argument('--use_clahe',
                        action='store_true',
                        default=True,
                        help='Use CLAHE preprocessing for better feature detection.')
    
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for LightGlue computation.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine the process based on the flags
    if args.use_split and args.use_connect:
        process = "Split+Connect_LightGlue"
    elif args.use_split:
        process = "Split_LightGlue"
    elif args.use_connect:
        process = "Connect_LightGlue"
    else:
        raise ValueError("Both use_split and use_connect are false, must at least use connect.")

    seq_tracks_dir = args.track_src
    data_path = os.path.dirname(seq_tracks_dir)
    seqs_tracks = os.listdir(seq_tracks_dir)
    
    tracker = args.tracker
    dataset = args.dataset

    seqs_tracks.sort()
    seq2Dist = dict()

    # Initialize LightGlue matcher
    logger.info("=" * 80)
    logger.info("Initializing LightGlue matcher...")
    logger.info(f"Video: {args.video_path}")
    logger.info(f"Max keypoints: {args.lightglue_max_keypoints}")
    logger.info(f"Samples per tracklet: {args.lightglue_samples}")
    logger.info(f"Sample strategy: {args.lightglue_sample_strategy}")
    logger.info("=" * 80)
    
    matcher = TrackletLightGlueMatcher(
        video_path=args.video_path,
        device=args.device,
        max_keypoints=args.lightglue_max_keypoints,
        confidence_threshold=args.lightglue_confidence,
        sample_strategy=args.lightglue_sample_strategy,
        max_samples_per_tracklet=args.lightglue_samples,
        enable_cache=True,
        cache_size_gb=args.lightglue_cache_size,
        use_clahe=args.use_clahe,
        batch_size=args.lightglue_batch_size
    )

    process_limit = 10000
    for seq_idx, seq in enumerate(seqs_tracks):
        if seq_idx >= process_limit:
            break
        
        seq_name = seq.split('.')[0]
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}: {seq_name}")
        
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)

        max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, args.spatial_factor)
        
        # Compute distance matrix using LightGlue
        Dist = get_distance_matrix_lightglue(tmp_trklets, matcher)
        seq2Dist[seq_name] = Dist
        display_Dist(Dist, seq_name, isMerged=False, isSplit=False)

        # Split if requested
        if args.use_split:
            logger.info(f"Number of tracklets before splitting: {len(tmp_trklets)}")
            splitTracklets = split_tracklets(tmp_trklets, eps=args.eps, max_k=args.max_k, 
                                            min_samples=args.min_samples, len_thres=args.min_len)
        else:
            splitTracklets = tmp_trklets
        
        # Recompute distance matrix after split
        Dist = get_distance_matrix_lightglue(splitTracklets, matcher)
        display_Dist(Dist, seq_name, isMerged=False, isSplit=True)
        logger.info(f"Number of tracklets before merging: {len(splitTracklets)}")
        
        # Merge tracklets
        mergedTracklets = merge_tracklets(splitTracklets, seq2Dist, Dist, seq_name=seq_name, 
                                         max_x_range=max_x_range, max_y_range=max_y_range, 
                                         merge_dist_thres=args.merge_dist_thres, matcher=matcher)
        
        Dist = get_distance_matrix_lightglue(mergedTracklets, matcher)
        display_Dist(Dist, seq_name, isMerged=True, isSplit=True)
        logger.info(f"Number of tracklets after merging: {len(mergedTracklets)}")

        # Save results
        sct_name = f'{tracker}_{dataset}_{process}_kp{args.lightglue_max_keypoints}_samples{args.lightglue_samples}_mergeDist{args.merge_dist_thres}'
        os.makedirs(os.path.join(data_path, sct_name), exist_ok=True)
        new_sct_output_path = os.path.join(data_path, sct_name, '{}.txt'.format(seq_name))
        save_results(new_sct_output_path, mergedTracklets)
        
        logger.info(f"Results saved to: {new_sct_output_path}")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
