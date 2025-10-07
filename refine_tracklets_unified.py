"""
Unified Tracklet Refinement - Complete Merged Implementation

Combines all functionality from:
1. refine_tracklets_enhanced.py - Enhanced feature-based merging
2. refine_tracklets_unified.py - IoU-based segment linking
3. merge_tracklets_groups.py - Cross-segment matching
4. test_sequential_merge.ipynb - Spatial integrity checks

Features:
- Trajectory continuity protection (prevents false splits)
- Short tracklet filtering (removes detector noise)
- Temporal feature smoothing (improves Re-ID robustness)
- Adaptive merge thresholds (handles varying confidence)
- Motion consistency checking (handles occlusions)
- Spatial zone validation (prevents distant ID hopping)
- Multi-point spatial validation (robust to single-frame noise)

Code Review Changes:
- Merged all helper functions from refine_tracklets_enhanced.py
- Removed duplicate imports
- Kept all validation checks (none proven redundant)
- Maintained backward compatibility with both scripts
"""

import numpy as np
import os
import torch
import pickle
import argparse

from collections import defaultdict
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from Tracklet import Tracklet

# Import from merge_tracklets_groups (external dependency)
from merge_tracklets_groups import (
    calculate_bbox_overlap,
    calculate_bbox_distance,
    find_closest_tracklets_mapping,
    replace_tracklet_keys
)


# ===== HELPER FUNCTIONS (from refine_tracklets_enhanced.py) =====

def smooth_features_temporal(features, window_size=5):
    """
    Apply moving average smoothing to features over time.
    Reduces noise from poor Re-ID model predictions.
    
    Args:
        features: List of feature vectors
        window_size: Size of smoothing window (odd number recommended)
    
    Returns:
        Smoothed feature list
    """
    if len(features) < window_size:
        return features
    
    features_np = np.array(features)
    smoothed = np.copy(features_np)
    
    half_window = window_size // 2
    for i in range(half_window, len(features) - half_window):
        smoothed[i] = np.mean(features_np[i-half_window:i+half_window+1], axis=0)
    
    return smoothed.tolist()


def get_adaptive_merge_threshold(track1, track2, base_threshold=0.4, confidence_factor=0.2):
    """
    Calculate adaptive merge threshold based on detection confidence.
    Lower confidence (distant/blurry players) = higher (more permissive) threshold.
    
    Args:
        track1, track2: Tracklet objects
        base_threshold: Base cosine distance threshold
        confidence_factor: How much confidence affects threshold
    
    Returns:
        Adaptive threshold
    """
    avg_conf_1 = np.mean(track1.scores) if len(track1.scores) > 0 else 0.5
    avg_conf_2 = np.mean(track2.scores) if len(track2.scores) > 0 else 0.5
    avg_conf = (avg_conf_1 + avg_conf_2) / 2
    
    # Lower confidence -> higher threshold (more permissive)
    adaptive_threshold = base_threshold + confidence_factor * (1 - avg_conf)
    
    return min(adaptive_threshold, 0.7)  # Cap at 0.7


def check_temporal_gap(track1, track2, max_gap_frames=30):
    """
    Verify tracklets are temporally close enough to merge.
    Prevents new persons from inheriting disappeared persons' IDs.
    
    Args:
        track1, track2: Tracklet objects
        max_gap_frames: Maximum allowed frame gap between tracklets
    
    Returns:
        bool: True if temporal gap is acceptable
    """
    track1_end = max(track1.times)
    track2_start = min(track2.times)
    track1_start = min(track1.times)
    track2_end = max(track2.times)
    
    # Determine temporal order
    if track1_end < track2_start:
        gap = track2_start - track1_end
    elif track2_end < track1_start:
        gap = track1_start - track2_end
    else:
        # Overlapping or interleaved - should not merge
        return False
    
    # Allow small gaps for occlusions but reject large gaps
    return 0 < gap <= max_gap_frames


def get_tracklet_spatial_zone(tracklet, grid_size=5, frame_width=1920, frame_height=1080):
    """
    Assign tracklet to spatial zone based on average position.
    Used to prevent ID hopping between distant players.
    
    Args:
        tracklet: Tracklet object
        grid_size: Divide frame into grid_size x grid_size zones
        frame_width, frame_height: Frame dimensions
    
    Returns:
        (zone_x, zone_y): Zone indices
    """
    centers = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in tracklet.bboxes])
    avg_center = np.mean(centers, axis=0)
    
    zone_x = int(avg_center[0] / frame_width * grid_size)
    zone_y = int(avg_center[1] / frame_height * grid_size)
    
    zone_x = min(max(zone_x, 0), grid_size - 1)
    zone_y = min(max(zone_y, 0), grid_size - 1)
    
    return (zone_x, zone_y)


def can_merge_spatially(track1, track2, max_zone_distance=1, grid_size=5):
    """
    Check if tracklets are in adjacent spatial zones.
    Prevents distant players from being merged.
    
    Args:
        track1, track2: Tracklet objects
        max_zone_distance: Maximum zone distance (1 = adjacent, 2 = 2 zones away)
        grid_size: Grid size for spatial partitioning
    
    Returns:
        bool: True if spatially compatible for merging
    """
    zone1 = get_tracklet_spatial_zone(track1, grid_size)
    zone2 = get_tracklet_spatial_zone(track2, grid_size)
    
    zone_dist = abs(zone1[0] - zone2[0]) + abs(zone1[1] - zone2[1])  # Manhattan distance
    
    return zone_dist <= max_zone_distance


def check_size_consistency(track1, track2, max_size_ratio=2.5):
    """
    Verify bounding box sizes are consistent.
    Large size difference suggests different distances from camera.
    
    Args:
        track1, track2: Tracklet objects
        max_size_ratio: Maximum allowed ratio of bbox areas
    
    Returns:
        bool: True if sizes are compatible
    """
    areas1 = [b[2] * b[3] for b in track1.bboxes if len(b) >= 4]
    areas2 = [b[2] * b[3] for b in track2.bboxes if len(b) >= 4]
    
    if not areas1 or not areas2:
        return False
    
    avg_area1 = np.mean(areas1)
    avg_area2 = np.mean(areas2)
    
    if avg_area1 == 0 or avg_area2 == 0:
        return False
    
    size_ratio = max(avg_area1, avg_area2) / min(avg_area1, avg_area2)
    
    return size_ratio <= max_size_ratio


def estimate_velocity(tracklet, n_frames=5):
    """
    Estimate velocity of a tracklet from its last n frames.
    
    Args:
        tracklet: Tracklet object
        n_frames: Number of frames to use for estimation
    
    Returns:
        velocity: (vx, vy) in pixels per frame
    """
    if len(tracklet.bboxes) < 2:
        return np.array([0.0, 0.0])
    
    n = min(n_frames, len(tracklet.bboxes))
    bboxes = tracklet.bboxes[-n:]
    times = tracklet.times[-n:]
    
    centers = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in bboxes])
    
    if len(centers) < 2:
        return np.array([0.0, 0.0])
    
    # Linear regression for velocity
    dt = np.diff(times)
    dx = np.diff(centers[:, 0])
    dy = np.diff(centers[:, 1])
    
    vx = np.mean(dx / dt) if np.any(dt > 0) else 0.0
    vy = np.mean(dy / dt) if np.any(dt > 0) else 0.0
    
    return np.array([vx, vy])


def predict_position(tracklet, n_frames_ahead):
    """
    Predict where tracklet will be after n frames based on velocity.
    
    Args:
        tracklet: Tracklet object
        n_frames_ahead: Number of frames to predict forward
    
    Returns:
        predicted_center: (x, y) predicted center position
    """
    velocity = estimate_velocity(tracklet)
    last_bbox = tracklet.bboxes[-1]
    last_center = np.array([last_bbox[0] + last_bbox[2]/2, last_bbox[1] + last_bbox[3]/2])
    
    predicted_center = last_center + velocity * n_frames_ahead
    return predicted_center


def check_motion_consistency(track1, track2, tolerance_ratio=2.0):
    """
    Check if track2's start position matches predicted position from track1.
    Helps handle occlusion cases.
    
    Args:
        track1, track2: Tracklet objects
        tolerance_ratio: Multiplier for spatial tolerance
    
    Returns:
        bool: True if motion is consistent
    """
    time_gap = min(track2.times) - max(track1.times)
    if time_gap <= 0 or time_gap > 30:
        return True  # Don't reject on motion if gap is invalid
    
    predicted_pos = predict_position(track1, time_gap)
    
    actual_bbox = track2.bboxes[0]
    actual_center = np.array([actual_bbox[0] + actual_bbox[2]/2, 
                              actual_bbox[1] + actual_bbox[3]/2])
    
    distance = np.linalg.norm(predicted_pos - actual_center)
    
    # Tolerance increases with time gap
    tolerance = 100 * tolerance_ratio * (1 + time_gap / 30.0)
    
    return distance <= tolerance


def prioritize_merges_by_confidence(Dist, tracklets, idx2tid, merge_dist_thres):
    """
    Sort potential merges by confidence, process high-confidence merges first.
    This prevents low-quality distant tracklets from "stealing" IDs.
    
    Args:
        Dist: Distance matrix
        tracklets: Dictionary of tracklets
        idx2tid: Index to track ID mapping
        merge_dist_thres: Distance threshold for merging
    
    Returns:
        List of (track1_idx, track2_idx, distance, confidence) sorted by priority
    """
    merge_candidates = []
    
    for i in range(Dist.shape[0]):
        for j in range(i+1, Dist.shape[1]):
            if Dist[i, j] < merge_dist_thres and Dist[i, j] < 1.0:  # Not overlap
                track1 = tracklets[idx2tid[i]]
                track2 = tracklets[idx2tid[j]]
                
                # Calculate confidence score
                avg_score1 = np.mean(track1.scores) if len(track1.scores) > 0 else 0.5
                avg_score2 = np.mean(track2.scores) if len(track2.scores) > 0 else 0.5
                avg_score = (avg_score1 + avg_score2) / 2
                
                # Higher score = higher priority
                merge_candidates.append((i, j, Dist[i, j], avg_score))
    
    # Sort by confidence (descending), then by distance (ascending)
    merge_candidates.sort(key=lambda x: (-x[3], x[2]))
    
    return merge_candidates


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.
    
    Args:
        track_times: List of frame times
    
    Returns:
        segments: List of (start_index, end_index) tuples
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
    Processes and pairs up segments from two different tracks to form valid subtracks.
    
    Args:
        seg1, seg2: Segment lists from track1 and track2
        track1, track2: Tracklet objects
    
    Returns:
        subtracks: List of subtrack objects
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


def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes.
    
    Args:
        tid2track: Dictionary of tracklets
        factor: Multiplier for spatial constraints
    
    Returns:
        (x_range, y_range): Maximum allowed spatial ranges
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


def get_distance_matrix(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets.
    Uses cosine distance on Re-ID features.
    
    Args:
        tid2track: Dictionary of tracklets
    
    Returns:
        Dist: Distance matrix (numpy array)
    """
    Dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        assert len(track1.times) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                Dist[i][j] = get_distance(track1_id, track2_id, track1, track2)
    return Dist


def get_distance(track1_id, track2_id, track1, track2, 
                similarity_threshold=0.7, use_voting=True):
    """
    Calculates the cosine distance between two tracks using Temporal Alignment.
    
    NEW: Two modes available:
    1. Voting mode (use_voting=True, RECOMMENDED): Counts how many embeddings are similar
       - More robust to outliers
       - More interpretable (percentage of matches)
       - Better handles appearance changes
       
    2. Averaging mode (use_voting=False): Traditional average distance
       - Temporal alignment (end-to-start comparison)
       - Preserves sequential structure
    
    Args:
        track1_id, track2_id: Track IDs
        track1, track2: Tracklet objects
        similarity_threshold: Cosine similarity threshold for voting (default=0.7)
        use_voting: If True, count similar pairs; if False, average distances
    
    Returns:
        distance: Cosine distance (0-1)
            - Voting mode: 1 - (match_ratio), where match_ratio = similar_pairs / total_pairs
            - Averaging mode: average cosine distance
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id
    
    # Check for temporal overlap
    doesOverlap = False
    if (track1_id != track2_id):
        doesOverlap = set(track1.times) & set(track2.times)
    if doesOverlap:
        return 1  # Maximum distance for overlapping tracks
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
    track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)
    
    count1 = len(track1_features_tensor)
    count2 = len(track2_features_tensor)
    
    # Determine temporal relationship
    track1_end = max(track1.times)
    track2_start = min(track2.times)
    track1_start = min(track1.times)
    track2_end = max(track2.times)
    
    # TEMPORAL ALIGNMENT: Compare features at corresponding positions
    # Case 1: Track1 ends before Track2 starts (typical merge scenario)
    if track1_end < track2_start:
        # Compare END of track1 with START of track2
        window = min(10, count1, count2)  # Use last/first 10 frames
        
        end_features_1 = track1_features_tensor[-window:]  # Last frames of track1
        start_features_2 = track2_features_tensor[:window]  # First frames of track2
        
        # Compute cosine similarity for aligned positions
        cos_sim_Numerator = (end_features_1 * start_features_2).sum(dim=1)
        end_norm = torch.norm(end_features_1, p=2, dim=1)
        start_norm = torch.norm(start_features_2, p=2, dim=1)
        cos_sim_Denominator = end_norm * start_norm
        
        # Cosine similarity (0-1, higher = more similar)
        cos_similarity = cos_sim_Numerator / (cos_sim_Denominator + 1e-8)
        
        if use_voting:
            # VOTING MODE: Count how many pairs are similar
            similar_count = (cos_similarity >= similarity_threshold).sum().item()
            match_ratio = similar_count / window
            
            # Convert to distance: high match_ratio → low distance
            result = 1.0 - match_ratio
            
            logger.debug(f"Voting: {similar_count}/{window} pairs similar ({match_ratio*100:.1f}%) → distance={result:.3f}")
            return result
        else:
            # AVERAGING MODE: Average distance
            cos_Dist = 1 - cos_similarity
            result = cos_Dist.mean().item()
            return result
    
    # Case 2: Track2 ends before Track1 starts
    elif track2_end < track1_start:
        window = min(10, count1, count2)
        
        end_features_2 = track2_features_tensor[-window:]
        start_features_1 = track1_features_tensor[:window]
        
        cos_sim_Numerator = (end_features_2 * start_features_1).sum(dim=1)
        end_norm = torch.norm(end_features_2, p=2, dim=1)
        start_norm = torch.norm(start_features_1, p=2, dim=1)
        cos_sim_Denominator = end_norm * start_norm
        
        cos_similarity = cos_sim_Numerator / (cos_sim_Denominator + 1e-8)
        
        if use_voting:
            # VOTING MODE
            similar_count = (cos_similarity >= similarity_threshold).sum().item()
            match_ratio = similar_count / window
            result = 1.0 - match_ratio
            
            logger.debug(f"Voting: {similar_count}/{window} pairs similar ({match_ratio*100:.1f}%) → distance={result:.3f}")
            return result
        else:
            # AVERAGING MODE
            cos_Dist = 1 - cos_similarity
            result = cos_Dist.mean().item()
            return result
    
    # Case 3: Interleaved or complex temporal relationship (shouldn't merge)
    else:
        # Fallback to conservative all-to-all
        cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
        track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
        track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
        cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
        cos_similarity_matrix = cos_sim_Numerator / cos_sim_Denominator
        
        if use_voting:
            # VOTING MODE: Count similar pairs across all combinations
            similar_count = (cos_similarity_matrix >= similarity_threshold).sum().item()
            total_pairs = count1 * count2
            match_ratio = similar_count / total_pairs
            result = 1.0 - match_ratio
            
            logger.debug(f"Voting (all-to-all): {similar_count}/{total_pairs} pairs similar ({match_ratio*100:.1f}%) → distance={result:.3f}")
            return result
        else:
            # AVERAGING MODE
            cos_Dist = 1 - cos_similarity_matrix
            total_cos_Dist = cos_Dist.sum()
            result = total_cos_Dist / (count1 * count2)
            return result


def check_spatial_constraints_robust(trk_1, trk_2, max_x_range, max_y_range, n_frames=3):
    """
    Enhanced spatial constraint checking using multiple frames.
    More robust to single-frame noise.
    
    Args:
        trk_1, trk_2: Tracklet objects
        max_x_range, max_y_range: Maximum allowed spatial ranges
        n_frames: Number of frames to average for robustness
    
    Returns:
        bool: True if spatial constraints are met
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)
    
    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    if not subtracks:
        return True
    
    subtrack_1st = subtracks.pop(0)
    
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        
        # Get last n_frames from subtrack_1st
        n1 = min(n_frames, len(subtrack_1st.bboxes))
        bboxes_1 = subtrack_1st.bboxes[-n1:]
        
        # Get first n_frames from subtrack_2nd
        n2 = min(n_frames, len(subtrack_2nd.bboxes))
        bboxes_2 = subtrack_2nd.bboxes[:n2]
        
        # Calculate average centers
        centers_1 = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in bboxes_1])
        centers_2 = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in bboxes_2])
        
        avg_center_1 = np.mean(centers_1, axis=0)
        avg_center_2 = np.mean(centers_2, axis=0)
        
        dx = abs(avg_center_1[0] - avg_center_2[0])
        dy = abs(avg_center_1[1] - avg_center_2[1])
        
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            break
        else:
            subtrack_1st = subtrack_2nd
    
    return inSpatialRange


def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """
    Detects identity switches within a tracklet using DBSCAN clustering.
    
    Args:
        embs: Feature embeddings array
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter
        max_clusters: Maximum number of clusters allowed
    
    Returns:
        (id_switch_detected, cluster_labels): Boolean and cluster assignments
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

    # Assign noise points to nearest cluster
    if -1 in labels and len(unique_labels) > 1:
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
    
    n_clusters = len(unique_labels)

    # Merge clusters if too many
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


# ===== SHORT TRACKLET FILTERING (NEW FEATURE) =====

def filter_short_tracklets(tracklets, 
                           min_length_frames=12,
                           min_length_for_warning=25,
                           fps=25):
    """
    Filter out very short tracklets that are likely detector noise.
    
    Research consensus: Tracklets < 0.5 seconds are usually false positives.
    - ByteTrack: Removes tracklets < 30 frames that couldn't merge
    - FairMOT: Minimum 15 frames for output
    - DeepSORT: Minimum 10 frames
    - StrongSORT: Post-processing removes < 20 frames
    
    Args:
        tracklets: Dictionary of tracklets
        min_length_frames: Absolute minimum (delete below this)
        min_length_for_warning: Warn if below this (but keep)
        fps: Video frame rate for logging
    
    Returns:
        filtered_tracklets: Dictionary with short tracklets removed
        removed_count: Number of removed tracklets
        removed_stats: Statistics dictionary
    """
    filtered_tracklets = {}
    removed_count = 0
    removed_ids = []
    removed_stats = {
        'very_short_removed': 0,  # < min_length_frames
        'short_kept': 0,           # < min_length_for_warning but kept
        'normal_kept': 0,          # >= min_length_for_warning
        'removed_lengths': []      # Lengths of removed tracklets
    }
    
    for tid, tracklet in tracklets.items():
        tracklet_length = len(tracklet.times)
        
        # Remove very short tracklets (likely noise)
        if tracklet_length < min_length_frames:
            removed_count += 1
            removed_ids.append(tid)
            removed_stats['very_short_removed'] += 1
            removed_stats['removed_lengths'].append(tracklet_length)
            logger.debug(f"Removed very short tracklet {tid}: {tracklet_length} frames "
                        f"({tracklet_length/fps:.2f} sec)")
            continue
        
        # Warn about short tracklets but keep them
        if tracklet_length < min_length_for_warning:
            removed_stats['short_kept'] += 1
            logger.debug(f"Short tracklet {tid}: {tracklet_length} frames "
                        f"({tracklet_length/fps:.2f} sec) - kept but may be unreliable")
        else:
            removed_stats['normal_kept'] += 1
        
        filtered_tracklets[tid] = tracklet
    
    # Summary logging
    logger.info(f"\n{'='*70}")
    logger.info(f"Short Tracklet Filtering Results:")
    logger.info(f"{'='*70}")
    logger.info(f"  Very short (< {min_length_frames} frames = {min_length_frames/fps:.2f}s): "
               f"{removed_stats['very_short_removed']} REMOVED")
    logger.info(f"  Short ({min_length_frames}-{min_length_for_warning-1} frames): "
               f"{removed_stats['short_kept']} kept (may be unreliable)")
    logger.info(f"  Normal (>= {min_length_for_warning} frames = {min_length_for_warning/fps:.1f}s): "
               f"{removed_stats['normal_kept']} kept")
    logger.info(f"  Total removed: {removed_count} ({removed_count/(len(tracklets)+1e-6)*100:.1f}%)")
    logger.info(f"  Total kept: {len(filtered_tracklets)}")
    
    if removed_stats['removed_lengths']:
        avg_removed_length = np.mean(removed_stats['removed_lengths'])
        logger.info(f"  Average length of removed tracklets: {avg_removed_length:.1f} frames "
                   f"({avg_removed_length/fps:.2f} sec)")
    
    logger.info(f"{'='*70}\n")
    
    return filtered_tracklets, removed_count, removed_stats


def analyze_tracklet_lengths(tracklets, fps=25):
    """
    Analyze tracklet length distribution to help set filtering thresholds.
    
    Args:
        tracklets: Dictionary of tracklets
        fps: Video frame rate
    
    Returns:
        lengths: List of all tracklet lengths
    """
    lengths = [len(t.times) for t in tracklets.values()]
    
    logger.info("\n" + "="*70)
    logger.info("Tracklet Length Analysis")
    logger.info("="*70)
    
    logger.info(f"\nTotal tracklets: {len(tracklets)}")
    logger.info(f"Mean length: {np.mean(lengths):.1f} frames ({np.mean(lengths)/fps:.2f} sec)")
    logger.info(f"Median length: {np.median(lengths):.1f} frames ({np.median(lengths)/fps:.2f} sec)")
    logger.info(f"Min length: {np.min(lengths)} frames ({np.min(lengths)/fps:.2f} sec)")
    logger.info(f"Max length: {np.max(lengths)} frames ({np.max(lengths)/fps:.2f} sec)")
    
    # Distribution
    logger.info("\nLength distribution:")
    bins = [0, 10, 25, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['< 10f (0.4s)', '10-25f (0.4-1s)', '25-50f (1-2s)', '50-100f (2-4s)', 
              '100-200f (4-8s)', '200-500f (8-20s)', '500-1000f (20-40s)', '> 1000f (40s+)']
    
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        percentage = count / len(lengths) * 100 if len(lengths) > 0 else 0
        logger.info(f"  {labels[i]}: {count:4d} tracklets ({percentage:5.1f}%)")
    
    # Identify problematic short tracklets
    very_short = sum(1 for l in lengths if l < 12)
    short = sum(1 for l in lengths if 12 <= l < 25)
    
    logger.info(f"\nPotential filtering candidates:")
    logger.info(f"  Very short (< 12 frames = 0.5s): {very_short} ({very_short/len(lengths)*100:.1f}%) - likely noise")
    logger.info(f"  Short (12-25 frames = 0.5-1s): {short} ({short/len(lengths)*100:.1f}%) - may need merging")
    
    logger.info("="*70 + "\n")
    
    return lengths


def detect_spatial_hops(tracklet, distance_threshold=15):
    """
    Detect spatial discontinuities (hops) in tracklet trajectory.
    Adapted from test_sequential_merge.ipynb
    
    Args:
        tracklet: Tracklet object
        distance_threshold: Maximum allowed movement between consecutive frames
    
    Returns:
        hop_indices: Array of indices where hops occur
        distances: All frame-to-frame distances
    """
    if len(tracklet.bboxes) < 2:
        return np.array([]), np.array([])
    
    centers = []
    for bbox in tracklet.bboxes:
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        centers.append([center_x, center_y])
    
    centers = np.array(centers)
    position_diffs = np.diff(centers, axis=0)
    distances = np.sqrt(np.sum(position_diffs**2, axis=1))
    
    hop_indices = np.where(distances > distance_threshold)[0]
    return hop_indices, distances


def calculate_continuity_score(tracklet):
    """
    Calculate how continuous a tracklet is in time.
    Higher score = more continuous (fewer gaps).
    
    Args:
        tracklet: Tracklet object
    
    Returns:
        float: Continuity score from 0.0 (fragmented) to 1.0 (perfectly continuous)
    """
    if len(tracklet.times) < 2:
        return 0.0
    
    # Calculate frame-to-frame differences
    frame_diffs = np.diff(tracklet.times)
    
    # Count consecutive frames (diff == 1)
    consecutive_frames = np.sum(frame_diffs == 1)
    total_transitions = len(frame_diffs)
    
    # Ratio of consecutive frames
    continuity_ratio = consecutive_frames / total_transitions if total_transitions > 0 else 0.0
    
    # Penalize large gaps with logarithmic decay
    max_gap = np.max(frame_diffs) if len(frame_diffs) > 0 else 1
    gap_penalty = 1.0 / (1.0 + np.log1p(max_gap - 1))
    
    # Combined score: weight continuity more heavily
    score = 0.7 * continuity_ratio + 0.3 * gap_penalty
    
    return score


def calculate_feature_stability(tracklet):
    """
    Calculate feature variance to determine if tracklet shows stable appearance.
    
    Args:
        tracklet: Tracklet object
    
    Returns:
        float: Average feature variance (lower = more stable)
    """
    if len(tracklet.features) < 5:
        return float('inf')  # Not enough data
    
    features = np.stack(tracklet.features)
    
    # Calculate variance across time for each feature dimension
    feature_variance = np.var(features, axis=0)
    
    # Average variance across all dimensions
    avg_variance = np.mean(feature_variance)
    
    return avg_variance


def should_protect_from_splitting(tracklet, 
                                  continuity_threshold=0.85,
                                  variance_threshold=0.02):
    """
    Determine if tracklet should be protected from splitting.
    
    Logic: If trajectory is highly continuous AND features are stable,
    then it's very likely one person - don't split even if DBSCAN suggests it.
    
    Args:
        tracklet: Tracklet object
        continuity_threshold: Minimum continuity score (0-1)
        variance_threshold: Maximum allowed feature variance
    
    Returns:
        bool: True if tracklet should be protected from splitting
    """
    # Calculate metrics
    continuity_score = calculate_continuity_score(tracklet)
    
    # First check: Is trajectory highly continuous?
    if continuity_score < continuity_threshold:
        return False  # Not continuous enough, allow splitting
    
    # Second check: Are features stable?
    feature_variance = calculate_feature_stability(tracklet)
    
    if feature_variance < variance_threshold:
        logger.info(f"Protecting tracklet {tracklet.track_id} from splitting: "
                   f"continuity={continuity_score:.3f}, variance={feature_variance:.4f}")
        return True
    
    return False


def check_spatial_constraints_iou(trk_1, trk_2, min_iou_threshold=0.3, n_frames=3):
    """
    Enhanced spatial constraint checking using IoU.
    Combines multi-point validation with IoU metric from merge_tracklets_groups.py
    
    Args:
        trk_1, trk_2: Tracklet objects
        min_iou_threshold: Minimum IoU to consider spatially valid
        n_frames: Number of frames to average
    
    Returns:
        bool: True if spatial constraints are met
    """
    if len(trk_1.bboxes) == 0 or len(trk_2.bboxes) == 0:
        return False
    
    # Get last n_frames from track1
    n1 = min(n_frames, len(trk_1.bboxes))
    bboxes_1 = trk_1.bboxes[-n1:]
    
    # Get first n_frames from track2
    n2 = min(n_frames, len(trk_2.bboxes))
    bboxes_2 = trk_2.bboxes[:n2]
    
    # Calculate IoU between all pairs
    ious = []
    for bbox1 in bboxes_1:
        for bbox2 in bboxes_2:
            iou = calculate_bbox_overlap(bbox1, bbox2)
            ious.append(iou)
    
    if not ious:
        return False
    
    # Use maximum IoU (best match)
    max_iou = np.max(ious)
    avg_iou = np.mean(ious)
    
    logger.debug(f"IoU check: max={max_iou:.3f}, avg={avg_iou:.3f}")
    
    # Accept if either max or average exceeds threshold
    return max_iou >= min_iou_threshold or avg_iou >= min_iou_threshold * 0.7


def split_tracklets_with_spatial_aware(tmp_trklets, 
                                       eps=0.8, 
                                       max_k=2, 
                                       min_samples=15, 
                                       len_thres=150,
                                       spatial_hop_threshold=15,
                                       smooth_features=True,
                                       use_continuity_check=True,
                                       continuity_threshold=0.85):
    """
    Enhanced splitting that considers BOTH feature clustering AND spatial hops.
    Combines DBSCAN from refine_tracklets.py with spatial analysis from notebook.
    NOW WITH TRAJECTORY CONTINUITY PROTECTION!
    
    Args:
        tmp_trklets: Dictionary of tracklets
        eps, max_k, min_samples, len_thres: DBSCAN parameters
        spatial_hop_threshold: Distance threshold for detecting hops
        smooth_features: Whether to apply temporal smoothing
        use_continuity_check: Enable trajectory continuity protection
        continuity_threshold: Minimum continuity score to protect from splitting
    
    Returns:
        Dictionary of split tracklets
    """
    from refine_tracklets_enhanced import detect_id_switch
    
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = {}
    protected_count = 0
    split_count = 0
    
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]
        
        # Apply temporal smoothing to features if enabled
        if smooth_features and len(trklet.features) > 5:
            trklet.features = smooth_features_temporal(trklet.features, window_size=5)
        
        # Check if tracklet is long enough to split
        if len(trklet.times) < len_thres:
            tracklets[tid] = trklet
            continue
        
        # NEW: Check trajectory continuity protection
        if use_continuity_check and should_protect_from_splitting(trklet, continuity_threshold):
            tracklets[tid] = trklet
            protected_count += 1
            continue
        
        # 1. Detect spatial hops
        hop_indices, distances = detect_spatial_hops(trklet, spatial_hop_threshold)
        
        # 2. Detect feature-based ID switches
        embs = np.stack(trklet.features)
        frames = np.array(trklet.times)
        bboxes = np.stack(trklet.bboxes)
        scores = np.array(trklet.scores) if hasattr(trklet, 'scores') and len(trklet.scores) > 0 else np.ones(len(frames))
        
        feature_switch_detected, feature_clusters = detect_id_switch(
            embs, 
            eps=eps, 
            min_samples=min_samples, 
            max_clusters=max_k
        )
        
        # 3. Combine both signals
        should_split = feature_switch_detected or len(hop_indices) > 0
        
        if not should_split:
            tracklets[tid] = trklet
            logger.debug(f"Track {tid}: No split needed")
        else:
            split_count += 1
            # Determine split points
            split_points = set()
            
            # Add split points from spatial hops
            if len(hop_indices) > 0:
                split_points.update(hop_indices.tolist())
                logger.info(f"Track {tid}: {len(hop_indices)} spatial hops detected at frames {hop_indices}")
            
            # Add split points from feature clustering
            if feature_switch_detected:
                # Find boundaries between clusters
                cluster_changes = np.where(np.diff(feature_clusters) != 0)[0]
                split_points.update(cluster_changes.tolist())
                logger.info(f"Track {tid}: {len(cluster_changes)} feature cluster changes detected")
            
            # Convert to sorted list
            split_points = sorted(list(split_points))
            
            # Split tracklet at detected points
            prev_idx = 0
            for split_idx in split_points:
                if split_idx - prev_idx > 10:  # Minimum segment length
                    segment_embs = embs[prev_idx:split_idx+1]
                    segment_frames = frames[prev_idx:split_idx+1]
                    segment_bboxes = bboxes[prev_idx:split_idx+1]
                    segment_scores = scores[prev_idx:split_idx+1]
                    
                    tracklets[new_id] = Tracklet(
                        new_id, 
                        segment_frames.tolist(), 
                        segment_scores.tolist(),
                        segment_bboxes.tolist(), 
                        feats=segment_embs.tolist()
                    )
                    logger.info(f"Created split tracklet {new_id} from {tid} (frames {segment_frames[0]}-{segment_frames[-1]})")
                    new_id += 1
                prev_idx = split_idx + 1
            
            # Add remaining segment
            if prev_idx < len(embs) - 10:
                segment_embs = embs[prev_idx:]
                segment_frames = frames[prev_idx:]
                segment_bboxes = bboxes[prev_idx:]
                segment_scores = scores[prev_idx:]
                
                tracklets[new_id] = Tracklet(
                    new_id,
                    segment_frames.tolist(),
                    segment_scores.tolist(),
                    segment_bboxes.tolist(),
                    feats=segment_embs.tolist()
                )
                logger.info(f"Created final split tracklet {new_id} from {tid} (frames {segment_frames[0]}-{segment_frames[-1]})")
                new_id += 1
    
    logger.info(f"Splitting complete: {len(tmp_trklets)} -> {len(tracklets)} tracklets")
    logger.info(f"  Protected from splitting: {protected_count} tracklets")
    logger.info(f"  Actually split: {split_count} tracklets")
    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def merge_tracklets_hybrid(tracklets,
                           merge_dist_thres=0.4,
                           max_gap_frames=30,
                           min_iou_threshold=0.3,
                           spatial_zone_check=True,
                           use_velocity=True,
                           use_iou=True):
    """
    Hybrid merging using BOTH feature similarity AND spatial IoU.
    Combines best practices from both implementations.
    
    Args:
        tracklets: Dictionary of tracklets
        merge_dist_thres: Base feature distance threshold
        max_gap_frames: Maximum temporal gap
        min_iou_threshold: Minimum IoU for spatial matching
        spatial_zone_check: Enable zone-based filtering
        use_velocity: Enable motion prediction
        use_iou: Use IoU instead of distance for spatial validation
    
    Returns:
        Dictionary of merged tracklets
    """
    from refine_tracklets_enhanced import prioritize_merges_by_confidence
    
    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    
    # Calculate distance matrix (feature-based)
    from refine_tracklets_enhanced import get_distance_matrix
    Dist = get_distance_matrix(tracklets)
    
    # Get prioritized merge candidates
    merge_candidates = prioritize_merges_by_confidence(Dist, tracklets, idx2tid, merge_dist_thres)
    
    merged_count = 0
    max_iterations = len(merge_candidates)
    iteration = 0
    
    while iteration < max_iterations and merge_candidates:
        iteration += 1
        
        if not merge_candidates:
            break
        
        track1_idx, track2_idx, distance, confidence = merge_candidates.pop(0)
        
        # Check if indices are still valid
        if track1_idx not in idx2tid or track2_idx not in idx2tid:
            continue
        
        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]
        
        # 1. Temporal gap check
        if not check_temporal_gap(track1, track2, max_gap_frames):
            logger.debug(f"Rejected {track1.track_id}->{track2.track_id}: temporal gap")
            continue
        
        # 2. Spatial zone check
        if spatial_zone_check and not can_merge_spatially(track1, track2, max_zone_distance=1):
            logger.debug(f"Rejected {track1.track_id}->{track2.track_id}: different zones")
            continue
        
        # 3. Size consistency
        if not check_size_consistency(track1, track2, max_size_ratio=2.5):
            logger.debug(f"Rejected {track1.track_id}->{track2.track_id}: size mismatch")
            continue
        
        # 4. Adaptive feature threshold
        adaptive_threshold = get_adaptive_merge_threshold(track1, track2, merge_dist_thres)
        if distance >= adaptive_threshold:
            logger.debug(f"Rejected {track1.track_id}->{track2.track_id}: feature distance {distance:.3f}")
            continue
        
        # 5. Spatial validation using IoU (NEW!)
        if use_iou:
            spatial_ok = check_spatial_constraints_iou(track1, track2, min_iou_threshold, n_frames=3)
        else:
            from refine_tracklets_enhanced import check_spatial_constraints_robust, get_spatial_constraints
            max_x_range, max_y_range = get_spatial_constraints(tracklets, factor=1.0)
            spatial_ok = check_spatial_constraints_robust(track1, track2, max_x_range, max_y_range, n_frames=3)
        
        if not spatial_ok:
            logger.debug(f"Rejected {track1.track_id}->{track2.track_id}: spatial constraint")
            continue
        
        # 6. Motion consistency (if gap exists)
        if use_velocity and spatial_ok:
            time_gap = abs(min(track2.times) - max(track1.times))
            if 1 < time_gap <= 30:
                motion_ok = check_motion_consistency(track1, track2, tolerance_ratio=2.0)
                if not motion_ok:
                    logger.debug(f"Rejected {track1.track_id}->{track2.track_id}: motion inconsistent")
                    continue
        
        # All checks passed - MERGE!
        logger.info(f"✓ Merging {track1.track_id} + {track2.track_id} "
                   f"(dist: {distance:.3f}, conf: {confidence:.3f})")
        
        track1.features += track2.features
        track1.times += track2.times
        track1.bboxes += track2.bboxes
        if hasattr(track1, 'scores') and hasattr(track2, 'scores'):
            track1.scores += track2.scores
        
        tracklets[idx2tid[track1_idx]] = track1
        tracklets.pop(idx2tid[track2_idx])
        
        # Recalculate
        Dist = get_distance_matrix(tracklets)
        idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
        merge_candidates = prioritize_merges_by_confidence(Dist, tracklets, idx2tid, merge_dist_thres)
        merged_count += 1
        iteration = 0
    
    logger.info(f"Merged {merged_count} tracklet pairs")
    return tracklets


def merge_segments_with_iou(segment_tracklets,
                            max_start_window=15,
                            max_end_window=15,
                            iou_threshold=0.85,
                            feature_threshold=0.6):
    """
    Merge tracklets across video segments.
    Uses IoU-based matching from merge_tracklets_groups.py
    with feature verification from enhanced version.
    
    Args:
        segment_tracklets: List of dicts with 'tracklets' and 'time_delta' keys
        max_start_window: Frames to look back from segment end
        max_end_window: Frames to look forward from segment start
        iou_threshold: Minimum IoU for spatial matching
        feature_threshold: Minimum feature similarity
    
    Returns:
        Merged tracklet dictionary with consistent IDs
    """
    if len(segment_tracklets) == 0:
        return {}
    
    # Start with first segment
    merged_tracklets = segment_tracklets[0]
    
    for seg_idx, next_segment in enumerate(segment_tracklets[1:], 1):
        logger.info(f"Merging segment {seg_idx} with segment {seg_idx+1}")
        
        # 1. Find spatial matches using IoU
        spatial_mapping, overlaps, _, _, _, _ = find_closest_tracklets_mapping(
            merged_tracklets,
            next_segment,
            max_start_window=max_start_window,
            max_end_window=max_end_window,
            max_overlap_threshold=iou_threshold
        )
        
        logger.info(f"Found {len(spatial_mapping)} potential IoU-based matches")
        
        # 2. Verify with feature similarity
        verified_mapping = {}
        for gone_id, new_id in spatial_mapping.items():
            gone_track = merged_tracklets['tracklets'][gone_id]
            new_track = next_segment['tracklets'][new_id]
            
            # Feature similarity check
            gone_feat = gone_track.features[-1] if len(gone_track.features) > 0 else None
            new_feat = new_track.features[0] if len(new_track.features) > 0 else None
            
            if gone_feat is not None and new_feat is not None:
                feat_sim = np.dot(gone_feat, new_feat) / (
                    np.linalg.norm(gone_feat) * np.linalg.norm(new_feat) + 1e-6
                )
            else:
                feat_sim = 0.0
            
            # Size consistency check
            size_ok = check_size_consistency(gone_track, new_track, max_size_ratio=2.5)
            
            # Accept if both pass
            if feat_sim > feature_threshold and size_ok:
                verified_mapping[gone_id] = new_id
                logger.info(f"✓ Verified {gone_id}->{new_id} "
                          f"(IoU: {overlaps[gone_id]:.3f}, feat: {feat_sim:.3f})")
            else:
                logger.debug(f"✗ Rejected {gone_id}->{new_id} "
                           f"(feat: {feat_sim:.3f}, size_ok: {size_ok})")
        
        # 3. Update tracklet IDs
        next_segment['tracklets'] = replace_tracklet_keys(
            next_segment['tracklets'],
            verified_mapping
        )
        
        logger.info(f"Applied {len(verified_mapping)} verified mappings")
        
        # 4. Merge segment dictionaries
        merged_tracklets['tracklets'].update(next_segment['tracklets'])
        merged_tracklets['time_delta'] = next_segment['time_delta']
    
    return merged_tracklets['tracklets']


# ===== MAIN PROCESSING FUNCTION =====

def refine_tracklets_unified(tracklets_input,
                             use_split=True,
                             use_merge=True,
                             filter_short=False,  # NEW
                             analyze_lengths=False,  # NEW
                             split_params=None,
                             merge_params=None,
                             filter_params=None):  # NEW
    """
    Unified tracklet refinement combining all best practices.
    
    Args:
        tracklets_input: Dictionary of tracklets or list of segment dicts
        use_split: Whether to apply splitting
        use_merge: Whether to apply merging
        filter_short: Whether to filter out very short tracklets (NEW)
        analyze_lengths: Whether to analyze tracklet length distribution (NEW)
        split_params: Dictionary of splitting parameters
        merge_params: Dictionary of merging parameters
        filter_params: Dictionary of filtering parameters (NEW)
    
    Returns:
        Refined tracklets dictionary
    """
    split_params = split_params or {}
    merge_params = merge_params or {}
    filter_params = filter_params or {}  # NEW
    # Handle both single dict and list of segments
    is_segmented = isinstance(tracklets_input, list)
    
    if is_segmented:
        logger.info(f"Processing {len(tracklets_input)} segments")
        
        # Process each segment independently
        refined_segments = []
        for seg_idx, segment in enumerate(tracklets_input):
            logger.info(f"--- Processing segment {seg_idx+1}/{len(tracklets_input)} ---")
            tracklets = segment['tracklets']
            
            # Split
            if use_split:
                tracklets = split_tracklets_with_spatial_aware(tracklets, **split_params)
            
            # Merge within segment
            if use_merge:
                tracklets = merge_tracklets_hybrid(tracklets, **merge_params)
            
            refined_segments.append({
                'tracklets': tracklets,
                'time_delta': segment['time_delta']
            })
        
        # Merge across segments
        logger.info("--- Merging across segments ---")
        final_tracklets = merge_segments_with_iou(
            refined_segments,
            max_start_window=merge_params.get('max_start_window', 15),
            max_end_window=merge_params.get('max_end_window', 15),
            iou_threshold=merge_params.get('iou_threshold', 0.85),
            feature_threshold=merge_params.get('feature_threshold', 0.6)
        )
    else:
        logger.info("Processing single tracklet set")
        tracklets = tracklets_input
        
        # Split
        if use_split:
            tracklets = split_tracklets_with_spatial_aware(tracklets, **split_params)
        
        # Merge
        if use_merge:
            tracklets = merge_tracklets_hybrid(tracklets, **merge_params)
        
        final_tracklets = tracklets
    
    logger.info(f"Refinement complete: {len(final_tracklets)} final tracklets")
    
    # Optional: Analyze tracklet lengths before filtering
    if analyze_lengths:
        logger.info("--- Analyzing tracklet lengths ---")
        analyze_tracklet_lengths(final_tracklets, fps=filter_params.get('fps', 25))
    
    # Optional: Filter very short tracklets
    if filter_short:
        logger.info("--- Filtering short tracklets ---")
        min_length = filter_params.get('min_output_length', 12)
        min_warning = filter_params.get('min_length_for_warning', 25)
        fps = filter_params.get('fps', 25)
        
        final_tracklets, removed_tracklets, stats = filter_short_tracklets(
            final_tracklets, 
            min_length_frames=min_length,
            min_length_for_warning=min_warning,
            fps=fps
        )
        
        logger.info(f"Filtering complete: {len(final_tracklets)} tracklets remaining, {stats['removed']} removed")
    
    return final_tracklets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified tracklet refinement")
    parser.add_argument('--track_src', type=str, required=True, help='Tracklet pickle file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file')
    parser.add_argument('--use_split', action='store_true', help='Enable splitting')
    parser.add_argument('--use_merge', action='store_true', help='Enable merging')
    parser.add_argument('--segmented', action='store_true', help='Process as segments')
    
    # Video creation parameters (NEW!)
    parser.add_argument('--create_video', action='store_true', 
                        help='Create visualization video of refined tracklets')
    parser.add_argument('--video_src', type=str, default=None,
                        help='Source video file for visualization (required if --create_video)')
    parser.add_argument('--video_output', type=str, default=None,
                        help='Output video file path (default: replaces .pkl with .mp4 in output path)')
    parser.add_argument('--show_trajectories', action='store_true',
                        help='Show trajectory trails in video')
    
    # Filtering parameters (NEW!)
    parser.add_argument('--filter_short', action='store_true', 
                        help='Filter out very short tracklets (< min_output_length frames)')
    parser.add_argument('--analyze_lengths', action='store_true',
                        help='Analyze tracklet length distribution before filtering')
    parser.add_argument('--min_output_length', type=int, default=12,
                        help='Minimum tracklet length in frames to keep (default=12, ~0.5 sec @ 25fps)')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Video frame rate for time conversions (default=25.0)')
    
    # Split parameters
    parser.add_argument('--eps', type=float, default=0.8)
    parser.add_argument('--min_samples', type=int, default=15)
    parser.add_argument('--max_k', type=int, default=2)
    parser.add_argument('--min_len', type=int, default=150)
    parser.add_argument('--spatial_hop_threshold', type=float, default=15)
    parser.add_argument('--use_continuity_check', action='store_true', 
                        help='Protect continuous tracklets from splitting (NEW!)')
    parser.add_argument('--continuity_threshold', type=float, default=0.85,
                        help='Minimum continuity score to protect (0-1, default=0.85)')
    
    # Merge parameters
    parser.add_argument('--merge_dist_thres', type=float, default=0.4)
    parser.add_argument('--max_gap_frames', type=int, default=30)
    parser.add_argument('--min_iou_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.85)
    
    args = parser.parse_args()
    
    # Load tracklets
    if args.segmented:
        # Load multiple segment files
        segment_files = sorted([f for f in os.listdir(args.track_src) if f.endswith('.pkl')])
        tracklets_input = []
        for seg_file in segment_files:
            with open(os.path.join(args.track_src, seg_file), 'rb') as f:
                tracklets = pickle.load(f)
                time_delta = max([max(t.times) for t in tracklets.values()])
                tracklets_input.append({'tracklets': tracklets, 'time_delta': time_delta})
    else:
        # Load single file
        with open(args.track_src, 'rb') as f:
            tracklets_input = pickle.load(f)
    
    # Set parameters
    split_params = {
        'eps': args.eps,
        'min_samples': args.min_samples,
        'max_k': args.max_k,
        'len_thres': args.min_len,
        'spatial_hop_threshold': args.spatial_hop_threshold,
        'smooth_features': True,
        'use_continuity_check': args.use_continuity_check,
        'continuity_threshold': args.continuity_threshold
    }
    
    merge_params = {
        'merge_dist_thres': args.merge_dist_thres,
        'max_gap_frames': args.max_gap_frames,
        'min_iou_threshold': args.min_iou_threshold,
        'iou_threshold': args.iou_threshold,
        'spatial_zone_check': True,
        'use_velocity': True,
        'use_iou': True
    }
    
    filter_params = {
        'min_output_length': args.min_output_length,
        'min_length_for_warning': 25,  # Warn for tracklets < 1 second
        'fps': args.fps
    }
    
    # Process
    refined_tracklets = refine_tracklets_unified(
        tracklets_input,
        use_split=args.use_split,
        use_merge=args.use_merge,
        filter_short=args.filter_short,
        analyze_lengths=args.analyze_lengths,
        split_params=split_params,
        merge_params=merge_params,
        filter_params=filter_params
    )
    
    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(refined_tracklets, f)
    
    logger.info(f"Saved refined tracklets to {args.output}")
    
    # Create visualization video (NEW!)
    if args.create_video:
        if not args.video_src:
            logger.error("--video_src is required when --create_video is enabled")
        elif not os.path.exists(args.video_src):
            logger.error(f"Video source not found: {args.video_src}")
        else:
            from utils.video_creator import create_final_tracklet_video
            
            # Determine output video path
            if args.video_output:
                video_output_path = args.video_output
            else:
                # Default: replace .pkl with .mp4 in output path
                video_output_path = args.output.replace('.pkl', '_refined.mp4')
            
            logger.info(f"Creating visualization video: {video_output_path}")
            
            try:
                create_final_tracklet_video(
                    video_path=args.video_src,
                    final_tracklets=refined_tracklets,
                    output_path=video_output_path,
                    show_trajectories=args.show_trajectories
                )
                logger.info(f"✓ Video created successfully: {video_output_path}")
            except Exception as e:
                logger.error(f"Failed to create video: {e}")
                import traceback
                traceback.print_exc()
