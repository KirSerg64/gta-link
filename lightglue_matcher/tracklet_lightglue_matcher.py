"""
Tracklet Distance Calculation using LightGlue Keypoint Matching

This module provides image-based tracklet matching using SuperPoint + LightGlue
for maximum quality tracklet association. Optimized for A100 GPU.

Key Features:
- Extracts player crops from video frames
- Uses SuperPoint + LightGlue for robust keypoint matching
- Intelligent frame sampling to balance quality and speed
- GPU-accelerated batch processing
- Feature caching to avoid redundant computations

Usage:
    from tracklet_lightglue_matcher import TrackletLightGlueMatcher
    
    matcher = TrackletLightGlueMatcher(
        video_path="path/to/video.mp4",
        device='cuda',
        max_keypoints=2048,
        confidence_threshold=0.3
    )
    
    distance = matcher.compute_distance(tracklet1, tracklet2)

Author: AI Assistant
Date: 2025-10-05
"""

import cv2 as cv
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
import pickle
from loguru import logger

# Import LightGlue components
try:
    from lightglue import LightGlue, SuperPoint
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    logger.warning("LightGlue not available. Install with: pip install git+https://github.com/cvg/LightGlue.git")


@dataclass
class MatchStatistics:
    """Statistics for keypoint matches between two tracklets"""
    num_matches: int
    avg_confidence: float
    match_density: float  # matches per frame pair
    spatial_consistency: float  # geometric consistency score
    distance: float  # final normalized distance [0, 1]


class FrameCache:
    """LRU cache for extracted frames and features to avoid redundant I/O"""
    
    def __init__(self, max_size_gb: float = 10.0):
        """
        Args:
            max_size_gb: Maximum cache size in GB
        """
        self.cache = {}
        self.access_order = []  # Track access order for LRU
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.current_size = 0
        self._eviction_count = 0
    
    def _get_key(self, video_path: str, frame_id: int, bbox: List[float]) -> str:
        """Generate unique key for frame crop"""
        key_str = f"{video_path}_{frame_id}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, video_path: str, frame_id: int, bbox: List[float]) -> Optional[np.ndarray]:
        """Retrieve cached frame crop (LRU)"""
        key = self._get_key(video_path, frame_id, bbox)
        if key in self.cache:
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, video_path: str, frame_id: int, bbox: List[float], crop: np.ndarray):
        """Store frame crop in cache with LRU eviction"""
        key = self._get_key(video_path, frame_id, bbox)
        crop_size = crop.nbytes
        
        # Evict least recently used items if needed
        while self.current_size + crop_size > self.max_size_bytes and self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                evicted_crop = self.cache.pop(lru_key)
                self.current_size -= evicted_crop.nbytes
                self._eviction_count += 1
        
        # If single item too large, skip caching
        if crop_size > self.max_size_bytes:
            logger.warning(f"Crop size {crop_size / 1024**2:.1f}MB exceeds cache limit, skipping")
            return
        
        self.cache[key] = crop
        self.access_order.append(key)
        self.current_size += crop_size
    
    def clear(self):
        """Clear cache"""
        logger.info(f"Clearing frame cache (evictions: {self._eviction_count}, size: {self.current_size / 1024**3:.2f}GB)")
        self.cache.clear()
        self.access_order.clear()
        self.current_size = 0
        self._eviction_count = 0


class TrackletLightGlueMatcher:
    """
    Image-based tracklet matching using SuperPoint + LightGlue
    Optimized for A100 GPU with 40GB memory
    """
    
    def __init__(self,
                 video_path: str,
                 device: str = 'cuda',
                 max_keypoints: int = 2048,
                 confidence_threshold: float = 0.3,
                 sample_strategy: str = 'uniform',
                 max_samples_per_tracklet: int = 10,
                 min_crop_size: int = 64,
                 enable_cache: bool = True,
                 cache_size_gb: float = 10.0,
                 use_clahe: bool = True,
                 batch_size: int = 16,
                 match_batch_size: int = 32):
        """
        Initialize LightGlue matcher for tracklet distance calculation
        
        Args:
            video_path: Path to video file
            device: 'cuda' or 'cpu'
            max_keypoints: Maximum keypoints per image (higher = more accurate, slower)
            confidence_threshold: Minimum confidence for valid matches
            sample_strategy: 'uniform', 'adaptive', or 'endpoints'
            max_samples_per_tracklet: Maximum frames to sample from each tracklet
            min_crop_size: Minimum width/height for valid crop
            enable_cache: Enable frame caching
            cache_size_gb: Cache size limit in GB
            use_clahe: Apply CLAHE for better feature detection
            batch_size: Batch size for feature extraction
            match_batch_size: Batch size for LightGlue matching (higher = faster but more memory)
        """
        if not LIGHTGLUE_AVAILABLE:
            raise ImportError("LightGlue is not installed. Please install it first.")
        
        self.video_path = video_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.sample_strategy = sample_strategy
        self.max_samples = max_samples_per_tracklet
        self.min_crop_size = min_crop_size
        self.use_clahe = use_clahe
        self.batch_size = batch_size
        self.match_batch_size = match_batch_size
        
        # Initialize cache
        self.cache = FrameCache(max_size_gb=cache_size_gb) if enable_cache else None
        
        # Initialize video capture
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        
        logger.info(f"Video loaded: {video_path}")
        logger.info(f"Total frames: {self.total_frames}, FPS: {self.fps}")
        
        # Initialize CLAHE for preprocessing
        if self.use_clahe:
            self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Initialize LightGlue models
        logger.info(f"Initializing LightGlue on {device} with max_keypoints={max_keypoints}")
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)
        
        logger.info("TrackletLightGlueMatcher initialized successfully")
    
    def __del__(self):
        """Release video capture"""
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def extract_frame_crop(self, frame_id: int, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extract and preprocess player crop from video frame
        
        Args:
            frame_id: Frame number
            bbox: [x, y, width, height]
        
        Returns:
            Preprocessed crop or None if invalid
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(self.video_path, frame_id, bbox)
            if cached is not None:
                return cached
        
        # Read frame from video
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_id}")
            return None
        
        # Extract bbox crop with padding
        x, y, w, h = map(int, bbox)
        
        # Add 20% padding to include context
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        
        # Validate crop size
        if crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
            logger.debug(f"Crop too small: {crop.shape}")
            return None
        
        # Preprocess: convert to grayscale and apply CLAHE
        if len(crop.shape) == 3:
            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        if self.use_clahe:
            gray = self.clahe.apply(gray)
        
        # Cache the crop
        if self.cache:
            self.cache.put(self.video_path, frame_id, bbox, gray)
        
        return gray
    
    def sample_frames(self, tracklet) -> List[Tuple[int, int, List[float]]]:
        """
        Sample representative frames from tracklet
        
        Args:
            tracklet: Tracklet object with times and bboxes
        
        Returns:
            List of (frame_id, index, bbox) tuples
        """
        n_frames = len(tracklet.times)
        
        if n_frames <= self.max_samples:
            # Use all frames
            return [(tracklet.times[i], i, tracklet.bboxes[i]) for i in range(n_frames)]
        
        # Sample based on strategy
        if self.sample_strategy == 'uniform':
            # Uniformly sample across tracklet
            indices = np.linspace(0, n_frames - 1, self.max_samples, dtype=int)
        
        elif self.sample_strategy == 'endpoints':
            # Sample more from start and end (identity more visible)
            n_start = self.max_samples // 3
            n_end = self.max_samples // 3
            n_middle = self.max_samples - n_start - n_end
            
            start_indices = np.linspace(0, n_frames // 4, n_start, dtype=int)
            end_indices = np.linspace(3 * n_frames // 4, n_frames - 1, n_end, dtype=int)
            middle_indices = np.linspace(n_frames // 4, 3 * n_frames // 4, n_middle, dtype=int)
            
            indices = np.concatenate([start_indices, middle_indices, end_indices])
            indices = np.unique(indices)
        
        elif self.sample_strategy == 'adaptive':
            # Sample based on bbox size (prefer larger, clearer detections)
            bbox_areas = [bbox[2] * bbox[3] for bbox in tracklet.bboxes]
            # Select frames with largest bboxes
            indices = np.argsort(bbox_areas)[-self.max_samples:]
            indices = np.sort(indices)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sample_strategy}")
        
        return [(tracklet.times[i], i, tracklet.bboxes[i]) for i in indices]
    
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor format expected by LightGlue
        
        Args:
            img: Grayscale image
        
        Returns:
            Tensor of shape (1, 1, H, W)
        """
        # Normalize to [0, 1]
        img_norm = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(img_norm)[None, None].to(self.device)
        
        return tensor
    
    @torch.no_grad()
    def extract_features_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Extract features from batch of images (GPU-optimized)
        
        Args:
            images: List of grayscale images
        
        Returns:
            List of feature dictionaries
        """
        if not images:
            return []
        
        features_list = []
        
        # Process in batches for efficiency
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Convert to tensors and stack
            tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                tensors.append(tensor)
            
            # Extract features for batch
            for tensor in tensors:
                features = self.extractor.extract(tensor)
                features_list.append(features)
        
        return features_list
    
    @torch.no_grad()
    def match_features(self,
                      features1: Dict,
                      features2: Dict,
                      img_shape1: Tuple[int, int],
                      img_shape2: Tuple[int, int]) -> Dict:
        """
        Match features between two images using LightGlue
        
        Returns:
            Dictionary with match information
        """
        # Add image shapes to features
        features1['image_size'] = torch.tensor(img_shape1)[None].to(self.device)
        features2['image_size'] = torch.tensor(img_shape2)[None].to(self.device)
        
        # Match features
        matches_dict = self.matcher({'image0': features1, 'image1': features2})
        
        # Extract matches
        matches = matches_dict['matches0'][0].cpu().numpy()
        confidence = matches_dict['matching_scores0'][0].cpu().numpy()
        
        # Filter by confidence
        valid = matches > -1
        valid_matches = matches[valid]
        valid_confidence = confidence[valid]
        
        # Apply confidence threshold
        high_conf_mask = valid_confidence >= self.confidence_threshold
        filtered_matches = valid_matches[high_conf_mask]
        filtered_confidence = valid_confidence[high_conf_mask]
        
        return {
            'num_matches': len(filtered_matches),
            'avg_confidence': np.mean(filtered_confidence) if len(filtered_confidence) > 0 else 0.0,
            'matches': filtered_matches,
            'confidence': filtered_confidence
        }
    
    @torch.no_grad()
    def match_features_batch(self,
                            features_pairs: List[Tuple[Dict, Dict, Tuple[int, int], Tuple[int, int]]],
                            batch_size: int = 32) -> List[Dict]:
        """
        Batch match multiple feature pairs simultaneously for better GPU utilization
        Includes robust memory management to prevent OOM errors.
        
        Args:
            features_pairs: List of (features1, features2, img_shape1, img_shape2) tuples
            batch_size: Number of pairs to process in parallel
        
        Returns:
            List of match result dictionaries
        """
        all_results = []
        
        # Process in batches to avoid OOM
        for batch_start in range(0, len(features_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(features_pairs))
            batch = features_pairs[batch_start:batch_end]
            
            try:
                # Prepare batch tensors
                batch_keypoints0 = []
                batch_descriptors0 = []
                batch_keypoints1 = []
                batch_descriptors1 = []
                batch_sizes0 = []
                batch_sizes1 = []
                
                for feat1, feat2, shape1, shape2 in batch:
                    # Extract tensors from feature dicts
                    batch_keypoints0.append(feat1['keypoints'][0])
                    batch_descriptors0.append(feat1['descriptors'][0])
                    batch_keypoints1.append(feat2['keypoints'][0])
                    batch_descriptors1.append(feat2['descriptors'][0])
                    batch_sizes0.append(torch.tensor(shape1, device=self.device))
                    batch_sizes1.append(torch.tensor(shape2, device=self.device))
                
                # Stack into batch tensors
                # Note: LightGlue expects [B, N, D] format
                batch_features0 = {
                    'keypoints': torch.stack(batch_keypoints0, dim=0),
                    'descriptors': torch.stack(batch_descriptors0, dim=0),
                    'image_size': torch.stack(batch_sizes0, dim=0)
                }
                batch_features1 = {
                    'keypoints': torch.stack(batch_keypoints1, dim=0),
                    'descriptors': torch.stack(batch_descriptors1, dim=0),
                    'image_size': torch.stack(batch_sizes1, dim=0)
                }
                
                # Perform batch matching
                matches_dict = self.matcher({'image0': batch_features0, 'image1': batch_features1})
                
                # Extract results for each pair in batch
                matches_batch = matches_dict['matches0'].cpu().numpy()  # [B, N]
                confidence_batch = matches_dict['matching_scores0'].cpu().numpy()  # [B, N]
                
                # Free GPU tensors immediately
                del batch_features0, batch_features1, matches_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for i in range(len(batch)):
                    matches = matches_batch[i]
                    confidence = confidence_batch[i]
                    
                    # Filter by confidence
                    valid = matches > -1
                    valid_matches = matches[valid]
                    valid_confidence = confidence[valid]
                    
                    # Apply confidence threshold
                    high_conf_mask = valid_confidence >= self.confidence_threshold
                    filtered_matches = valid_matches[high_conf_mask]
                    filtered_confidence = valid_confidence[high_conf_mask]
                    
                    all_results.append({
                        'num_matches': len(filtered_matches),
                        'avg_confidence': np.mean(filtered_confidence) if len(filtered_confidence) > 0 else 0.0,
                        'matches': filtered_matches,
                        'confidence': filtered_confidence
                    })
                
                # Clean up batch arrays
                del matches_batch, confidence_batch
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM error - try with smaller batch
                    logger.error(f"OOM error with batch size {len(batch)}, falling back to sequential processing")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process sequentially as fallback
                    for feat1, feat2, shape1, shape2 in batch:
                        try:
                            result = self.match_features(feat1, feat2, shape1, shape2)
                            all_results.append(result)
                        except Exception as e2:
                            logger.error(f"Failed to match features: {e2}")
                            all_results.append({
                                'num_matches': 0,
                                'avg_confidence': 0.0,
                                'matches': np.array([]),
                                'confidence': np.array([])
                            })
                else:
                    raise e
        
        return all_results
    
    def compute_tracklet_features(self, tracklet) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Extract crops and features for sampled frames from tracklet
        
        Returns:
            Tuple of (crops, features)
        """
        sampled_frames = self.sample_frames(tracklet)
        
        crops = []
        valid_frames = []
        
        # Extract all crops first
        for frame_id, idx, bbox in sampled_frames:
            crop = self.extract_frame_crop(frame_id, bbox)
            if crop is not None:
                crops.append(crop)
                valid_frames.append((frame_id, idx, bbox))
        
        if not crops:
            logger.warning(f"No valid crops extracted for tracklet {tracklet.track_id}")
            return [], []
        
        # Extract features in batch
        features = self.extract_features_batch(crops)
        
        return crops, features
    
    def compute_distance(self, track1, track2) -> float:
        """
        Compute distance between two tracklets using keypoint matching
        
        Args:
            track1: First tracklet
            track2: Second tracklet
        
        Returns:
            Distance value in [0, 1], where 0 = same identity, 1 = different
        """
        # Extract features for both tracklets
        crops1, features1 = self.compute_tracklet_features(track1)
        crops2, features2 = self.compute_tracklet_features(track2)
        
        if not features1 or not features2:
            logger.warning(f"Failed to extract features for tracklets {track1.track_id}, {track2.track_id}")
            return 1.0  # Maximum distance if no features
        
        # Match all pairs of frames between tracklets
        total_matches = 0
        total_confidence = 0.0
        num_pairs = 0
        
        for i, feat1 in enumerate(features1):
            for j, feat2 in enumerate(features2):
                match_result = self.match_features(
                    feat1,
                    feat2,
                    crops1[i].shape[:2],
                    crops2[j].shape[:2]
                )
                
                num_matches = match_result['num_matches']
                avg_confidence = match_result['avg_confidence']
                
                total_matches += num_matches
                total_confidence += avg_confidence * num_matches
                num_pairs += 1
        
        if total_matches == 0:
            # No matches found - likely different identities
            return 1.0
        
        # Calculate statistics
        avg_matches_per_pair = total_matches / num_pairs
        avg_confidence = total_confidence / total_matches
        
        # Normalize to distance [0, 1]
        # Good matches: 50+ matches with high confidence -> distance ~0
        # Poor matches: <10 matches with low confidence -> distance ~1
        
        # Score based on match count (more matches = lower distance)
        match_score = 1.0 / (1.0 + avg_matches_per_pair / 20.0)
        
        # Score based on confidence
        confidence_score = 1.0 - avg_confidence
        
        # Combined distance (weighted average)
        distance = 0.6 * match_score + 0.4 * confidence_score
        
        return float(distance)
    
    def compute_distance_matrix(self, tracklets: Dict) -> np.ndarray:
        """
        Compute full distance matrix for all tracklets (optimized with memory management)
        
        Args:
            tracklets: Dictionary of tracklet_id -> Tracklet
        
        Returns:
            Distance matrix [N x N]
        """
        n = len(tracklets)
        dist_matrix = np.zeros((n, n))
        
        tid_list = list(tracklets.keys())
        tid2idx = {tid: i for i, tid in enumerate(tid_list)}
        
        logger.info(f"Computing distance matrix for {n} tracklets using LightGlue")
        
        # Memory management: clear cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Pre-extract features for all tracklets (with caching)
        logger.info("Pre-extracting features for all tracklets...")
        tracklet_features = {}
        for tid_idx, tid in enumerate(tqdm(tid_list, desc="Extracting features")):
            crops, features = self.compute_tracklet_features(tracklets[tid])
            tracklet_features[tid] = (crops, features)
            
            # Memory management: periodically clear cache
            if tid_idx > 0 and tid_idx % 20 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Compute pairwise distances
        logger.info("Computing pairwise distances...")
        total_pairs = n * (n - 1) // 2
        
        with tqdm(total=total_pairs, desc="Matching tracklets") as pbar:
            for i, tid1 in enumerate(tid_list):
                crops1, features1 = tracklet_features[tid1]
                
                for j, tid2 in enumerate(tid_list):
                    if j <= i:
                        if j == i:
                            dist_matrix[i][j] = 0.0
                        else:
                            dist_matrix[i][j] = dist_matrix[j][i]
                        continue
                    
                    # Check for temporal overlap
                    track1 = tracklets[tid1]
                    track2 = tracklets[tid2]
                    
                    if set(track1.times) & set(track2.times):
                        # Overlapping tracks - maximum distance
                        dist_matrix[i][j] = 1.0
                        pbar.update(1)
                        continue
                    
                    # Compute distance
                    crops2, features2 = tracklet_features[tid2]
                    
                    if not features1 or not features2:
                        dist_matrix[i][j] = 1.0
                        pbar.update(1)
                        continue
                    
                    # Build batch of all frame pairs for this tracklet pair
                    frame_pairs = []
                    for feat1, crop1 in zip(features1, crops1):
                        for feat2, crop2 in zip(features2, crops2):
                            frame_pairs.append((feat1, feat2, crop1.shape[:2], crop2.shape[:2]))
                    
                    # Batch match all frame pairs at once (VECTORIZED!)
                    match_results = self.match_features_batch(frame_pairs, batch_size=self.match_batch_size)
                    
                    # Aggregate results
                    total_matches = sum(r['num_matches'] for r in match_results)
                    total_confidence = sum(r['avg_confidence'] * r['num_matches'] for r in match_results)
                    num_pairs = len(match_results)
                    
                    if total_matches == 0:
                        dist_matrix[i][j] = 1.0
                    else:
                        avg_matches_per_pair = total_matches / num_pairs
                        avg_confidence = total_confidence / total_matches
                        
                        match_score = 1.0 / (1.0 + avg_matches_per_pair / 20.0)
                        confidence_score = 1.0 - avg_confidence
                        distance = 0.6 * match_score + 0.4 * confidence_score
                        dist_matrix[i][j] = distance
                    
                    pbar.update(1)
                    
                    # Memory management: periodically clear cache
                    if pbar.n % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Distance matrix computation complete")
        return dist_matrix


def get_distance_lightglue(track1_id, track2_id, track1, track2, matcher: TrackletLightGlueMatcher) -> float:
    """
    Wrapper function compatible with refine_tracklets.py
    
    Args:
        track1_id: ID of first tracklet
        track2_id: ID of second tracklet
        track1: First tracklet object
        track2: Second tracklet object
        matcher: TrackletLightGlueMatcher instance
    
    Returns:
        Distance value [0, 1]
    """
    # Check for temporal overlap
    if track1_id != track2_id:
        if set(track1.times) & set(track2.times):
            return 1.0
    
    # Compute keypoint-based distance
    distance = matcher.compute_distance(track1, track2)
    
    return distance
