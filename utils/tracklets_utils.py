from collections import defaultdict
import os
import cv2
from loguru import logger
from tqdm import tqdm


def extract_tracklet_crops(video_path, tracklets, output_dir=None, crop_size=(128, 256), padding=0):
    """
    Extract and save bbox crops for each tracklet over time for analysis.
    
    Args:
        video_path (str): Path to the input video
        tracklets (dict): Dictionary of tracklets {tracklet_id: Tracklet}
        output_dir (str): Directory to save crops. If None, creates 'tracklet_crops' in video directory
        crop_size (tuple): Target size for crops (width, height). If None, uses original bbox size
        padding (int): Extra pixels around bbox for context
    """
    if output_dir is None:
        video_dir, video_name = os.path.split(video_path)
        video_name = os.path.splitext(video_name)[0]
        output_dir = os.path.join(video_dir, f"{video_name}_tracklet_crops")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Extracting crops from video: {video_path}")
    logger.info(f"Video dimensions: {frame_width}x{frame_height}, Total frames: {total_frames}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create a mapping from frame number to tracklets present in that frame
    frame_to_tracklets = defaultdict(list)
    for tid, tracklet in tracklets.items():
        for i, frame_num in enumerate(tracklet.times):
            frame_to_tracklets[frame_num].append((tid, i))
    
    frame_num = 0
    progress_bar = tqdm(total=total_frames, desc="Extracting tracklet crops")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process tracklets in current frame
        if frame_num in frame_to_tracklets:
            for tid, bbox_idx in frame_to_tracklets[frame_num]:
                tracklet = tracklets[tid]
                bbox = tracklet.bboxes[bbox_idx]
                
                # Extract bbox coordinates
                x, y, w, h = bbox[:4]
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Add padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame_width, x + w + padding)
                y2 = min(frame_height, y + h + padding)
                
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Resize if target size is specified
                if crop_size is not None:
                    crop = cv2.resize(crop, crop_size)
                
                # Create tracklet directory
                tracklet_dir = os.path.join(output_dir, f"tracklet_{tid:04d}")
                os.makedirs(tracklet_dir, exist_ok=True)
                
                # Save crop with frame number and bbox info
                crop_filename = f"{frame_num:06d}.jpg"
                crop_path = os.path.join(tracklet_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
        
        frame_num += 1
        progress_bar.update(1)
    
    progress_bar.close()
    cap.release()
    
    logger.info(f"Tracklet crops extraction completed. Saved to: {output_dir}")