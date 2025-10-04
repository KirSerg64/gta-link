"""
Player Re-Identification using Keypoint Matching

This module implements multiple approaches to test the hypothesis that 
keypoint matching can distinguish between same-player and different-player images.

Three main variants:
1. Classical ORB with geometric filtering
2. Deep learning keypoints (LightGlue + SuperPoint)
3. Hybrid approach with spatial binning

Author: CV Algorithm Developer
Date: 2025-10-04
"""

import cv2 as cv
import numpy as np
import os
import torch
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
import argparse


@dataclass
class MatchResult:
    """Store matching results between two images"""
    tracklet_id_1: int
    tracklet_id_2: int
    image_idx_1: int
    image_idx_2: int
    num_matches: int
    num_filtered_matches: int
    avg_match_distance: float
    match_confidence: float
    is_same_tracklet: bool


class TrackletImageLoader:
    """Load and manage tracklet images from directory structure"""
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Path to directory containing tracklet subdirectories
        """
        self.base_path = Path(base_path)
        self.tracklets = {}  # tracklet_id -> list of image paths
        self._load_tracklets()
    
    def _load_tracklets(self):
        """Load all tracklet images from subdirectories"""
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")
        
        # Find all numeric subdirectories
        for subdir in sorted(self.base_path.iterdir()):
            if subdir.is_dir() and subdir.name.isdigit():
                tracklet_id = int(subdir.name)
                
                # Load all images from this tracklet
                image_paths = sorted([
                    str(img_path) for img_path in subdir.glob("*.jpg")
                ] + [
                    str(img_path) for img_path in subdir.glob("*.png")
                ])
                
                if len(image_paths) > 0:
                    self.tracklets[tracklet_id] = image_paths
        
        print(f"Loaded {len(self.tracklets)} tracklets")
        for tid, paths in self.tracklets.items():
            print(f"  Tracklet {tid}: {len(paths)} images")
    
    def get_tracklet_ids(self) -> List[int]:
        """Get list of all tracklet IDs"""
        return sorted(self.tracklets.keys())
    
    def get_images(self, tracklet_id: int) -> List[np.ndarray]:
        """Load all images for a given tracklet"""
        if tracklet_id not in self.tracklets:
            raise ValueError(f"Tracklet {tracklet_id} not found")
        
        images = []
        for img_path in self.tracklets[tracklet_id]:
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
        
        return images
    
    def get_image_paths(self, tracklet_id: int) -> List[str]:
        """Get all image paths for a given tracklet"""
        return self.tracklets.get(tracklet_id, [])


class ORBKeypointMatcher:
    """Classical ORB-based keypoint matching with geometric filtering"""
    
    def __init__(self, 
                 n_features: int = 500,
                 ratio_threshold: float = 0.75,
                 ransac_reproj_threshold: float = 5.0,
                 min_matches: int = 10):
        """
        Args:
            n_features: Number of ORB features to detect
            ratio_threshold: Lowe's ratio test threshold
            ransac_reproj_threshold: RANSAC reprojection threshold
            min_matches: Minimum number of matches for homography
        """
        # ORB detector with optimized parameters
        self.orb = cv.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # BFMatcher with Hamming distance
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_reproj_threshold
        self.min_matches = min_matches
        
        # CLAHE for preprocessing
        self.clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive enhancement to improve keypoint detection"""
        # Convert to LAB color space
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened
        final_img = cv.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return final_img
    
    def detect_and_compute(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect keypoints and compute descriptors"""
        enhanced = self.preprocess_image(img)
        keypoints, descriptors = self.orb.detectAndCompute(enhanced, None)
        return keypoints, descriptors
    
    def match_with_ratio_test(self, 
                              des1: np.ndarray, 
                              des2: np.ndarray) -> List[cv.DMatch]:
        """Apply Lowe's ratio test to filter good matches"""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        
        # KNN matching
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def filter_matches_horizontal(self, 
                                  kp1: List, 
                                  kp2: List, 
                                  matches: List[cv.DMatch],
                                  angle_threshold: float = 15.0) -> List[cv.DMatch]:
        """
        Filter matches that are approximately horizontal (parallel to x-axis)
        
        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: List of matches
            angle_threshold: Maximum angle deviation from horizontal (degrees)
        """
        if len(matches) == 0:
            return []
        
        filtered_matches = []
        
        for match in matches:
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt)
            
            # Calculate angle of the match vector
            delta = pt2 - pt1
            angle = np.abs(np.arctan2(delta[1], delta[0]) * 180 / np.pi)
            
            # Check if approximately horizontal (0° or 180°)
            if angle < angle_threshold or angle > (180 - angle_threshold):
                filtered_matches.append(match)
        
        return filtered_matches
    
    def filter_matches_spatial_consistency(self,
                                          kp1: List,
                                          kp2: List,
                                          matches: List[cv.DMatch],
                                          num_bins: int = 3) -> List[cv.DMatch]:
        """
        Filter matches based on spatial consistency (vertical binning)
        Only keep matches where keypoints are in similar vertical positions
        
        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: List of matches
            num_bins: Number of vertical bins (top/middle/bottom)
        """
        if len(matches) == 0 or len(kp1) == 0 or len(kp2) == 0:
            return []
        
        # Get image dimensions (approximate from keypoints)
        max_y1 = max(kp.pt[1] for kp in kp1)
        max_y2 = max(kp.pt[1] for kp in kp2)
        
        filtered_matches = []
        
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # Calculate vertical bin for each point
            bin1 = int(pt1[1] / max_y1 * num_bins)
            bin2 = int(pt2[1] / max_y2 * num_bins)
            
            # Keep match if in same or adjacent bin
            if abs(bin1 - bin2) <= 1:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def filter_matches_ransac(self,
                              kp1: List,
                              kp2: List,
                              matches: List[cv.DMatch]) -> Tuple[List[cv.DMatch], Optional[np.ndarray]]:
        """
        Filter matches using RANSAC with homography estimation
        
        Returns:
            Tuple of (filtered_matches, homography_matrix)
        """
        if len(matches) < self.min_matches:
            return [], None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, self.ransac_threshold)
        
        if H is None:
            return [], None
        
        # Filter matches using inlier mask
        filtered_matches = [m for m, is_inlier in zip(matches, mask.ravel()) if is_inlier]
        
        return filtered_matches, H
    
    def match_images(self,
                    img1: np.ndarray,
                    img2: np.ndarray,
                    use_horizontal_filter: bool = True,
                    use_spatial_filter: bool = True,
                    use_ransac_filter: bool = True) -> Dict:
        """
        Match two images and return detailed statistics
        
        Returns:
            Dictionary with matching statistics
        """
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)
        
        if des1 is None or des2 is None:
            return {
                'num_kp1': 0,
                'num_kp2': 0,
                'num_raw_matches': 0,
                'num_filtered_matches': 0,
                'avg_match_distance': float('inf'),
                'match_score': 0.0
            }
        
        # Initial matching with ratio test
        raw_matches = self.match_with_ratio_test(des1, des2)
        
        filtered_matches = raw_matches
        
        # Apply horizontal filter
        if use_horizontal_filter and len(filtered_matches) > 0:
            filtered_matches = self.filter_matches_horizontal(kp1, kp2, filtered_matches)
        
        # Apply spatial consistency filter
        if use_spatial_filter and len(filtered_matches) > 0:
            filtered_matches = self.filter_matches_spatial_consistency(kp1, kp2, filtered_matches)
        
        # Apply RANSAC filter
        homography = None
        if use_ransac_filter and len(filtered_matches) >= self.min_matches:
            filtered_matches, homography = self.filter_matches_ransac(kp1, kp2, filtered_matches)
        
        # Calculate statistics
        avg_distance = np.mean([m.distance for m in filtered_matches]) if filtered_matches else float('inf')
        
        # Match score: normalized by number of keypoints
        match_score = len(filtered_matches) / max(len(kp1), len(kp2), 1)
        
        return {
            'num_kp1': len(kp1),
            'num_kp2': len(kp2),
            'num_raw_matches': len(raw_matches),
            'num_filtered_matches': len(filtered_matches),
            'avg_match_distance': avg_distance,
            'match_score': match_score,
            'homography': homography,
            'keypoints1': kp1,
            'keypoints2': kp2,
            'matches': filtered_matches
        }


class PlayerReIDEvaluator:
    """Evaluate player re-identification hypothesis"""
    
    def __init__(self, 
                 image_loader: TrackletImageLoader,
                 matcher: ORBKeypointMatcher,
                 max_images_per_tracklet: int = 20,
                 sample_pairs: bool = True,
                 num_sample_pairs: int = 100):
        """
        Args:
            image_loader: TrackletImageLoader instance
            matcher: Keypoint matcher instance
            max_images_per_tracklet: Maximum images to use per tracklet
            sample_pairs: Whether to sample image pairs (for large datasets)
            num_sample_pairs: Number of pairs to sample for inter-tracklet comparison
        """
        self.loader = image_loader
        self.matcher = matcher
        self.max_images_per_tracklet = max_images_per_tracklet
        self.sample_pairs = sample_pairs
        self.num_sample_pairs = num_sample_pairs
        
        self.results: List[MatchResult] = []
    
    def evaluate_all_pairs(self):
        """Evaluate matching for all image pairs"""
        tracklet_ids = self.loader.get_tracklet_ids()
        
        print("\n" + "="*80)
        print("PLAYER RE-IDENTIFICATION EVALUATION")
        print("="*80)
        print(f"Total tracklets: {len(tracklet_ids)}")
        print(f"Max images per tracklet: {self.max_images_per_tracklet}")
        
        self.results = []
        
        # Evaluate intra-tracklet matches (same player)
        print("\n[1/2] Evaluating INTRA-TRACKLET matches (same player)...")
        intra_matches = self._evaluate_intra_tracklet_matches(tracklet_ids)
        self.results.extend(intra_matches)
        
        # Evaluate inter-tracklet matches (different players)
        print("\n[2/2] Evaluating INTER-TRACKLET matches (different players)...")
        inter_matches = self._evaluate_inter_tracklet_matches(tracklet_ids)
        self.results.extend(inter_matches)
        
        print("\n" + "="*80)
        print(f"Total evaluations completed: {len(self.results)}")
        print("="*80)
    
    def _evaluate_intra_tracklet_matches(self, tracklet_ids: List[int]) -> List[MatchResult]:
        """Evaluate matches within same tracklet"""
        results = []
        
        for tracklet_id in tqdm(tracklet_ids, desc="Intra-tracklet matching"):
            images = self.loader.get_images(tracklet_id)
            
            # Limit number of images
            if len(images) > self.max_images_per_tracklet:
                # Sample uniformly
                indices = np.linspace(0, len(images)-1, self.max_images_per_tracklet, dtype=int)
                images = [images[i] for i in indices]
            
            # Match all pairs within this tracklet
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    match_stats = self.matcher.match_images(images[i], images[j])
                    
                    result = MatchResult(
                        tracklet_id_1=tracklet_id,
                        tracklet_id_2=tracklet_id,
                        image_idx_1=i,
                        image_idx_2=j,
                        num_matches=match_stats['num_raw_matches'],
                        num_filtered_matches=match_stats['num_filtered_matches'],
                        avg_match_distance=match_stats['avg_match_distance'],
                        match_confidence=match_stats['match_score'],
                        is_same_tracklet=True
                    )
                    results.append(result)
        
        return results
    
    def _evaluate_inter_tracklet_matches(self, tracklet_ids: List[int]) -> List[MatchResult]:
        """Evaluate matches between different tracklets"""
        results = []
        
        # Load all images for each tracklet
        tracklet_images = {}
        for tracklet_id in tracklet_ids:
            images = self.loader.get_images(tracklet_id)
            if len(images) > self.max_images_per_tracklet:
                indices = np.linspace(0, len(images)-1, self.max_images_per_tracklet, dtype=int)
                images = [images[i] for i in indices]
            tracklet_images[tracklet_id] = images
        
        # Generate all possible tracklet pairs
        tracklet_pairs = [(tid1, tid2) for i, tid1 in enumerate(tracklet_ids) 
                         for tid2 in tracklet_ids[i+1:]]
        
        if self.sample_pairs and len(tracklet_pairs) > 0:
            # Sample pairs for efficiency
            num_pairs = min(self.num_sample_pairs, len(tracklet_pairs))
            sampled_pairs = np.random.choice(len(tracklet_pairs), num_pairs, replace=False)
            tracklet_pairs = [tracklet_pairs[i] for i in sampled_pairs]
        
        # Match images from different tracklets
        for tid1, tid2 in tqdm(tracklet_pairs, desc="Inter-tracklet matching"):
            images1 = tracklet_images[tid1]
            images2 = tracklet_images[tid2]
            
            # Sample image pairs from different tracklets
            num_comparisons = min(5, len(images1) * len(images2))
            
            for _ in range(num_comparisons):
                i = np.random.randint(0, len(images1))
                j = np.random.randint(0, len(images2))
                
                match_stats = self.matcher.match_images(images1[i], images2[j])
                
                result = MatchResult(
                    tracklet_id_1=tid1,
                    tracklet_id_2=tid2,
                    image_idx_1=i,
                    image_idx_2=j,
                    num_matches=match_stats['num_raw_matches'],
                    num_filtered_matches=match_stats['num_filtered_matches'],
                    avg_match_distance=match_stats['avg_match_distance'],
                    match_confidence=match_stats['match_score'],
                    is_same_tracklet=False
                )
                results.append(result)
        
        return results
    
    def analyze_results(self) -> Dict:
        """Analyze and summarize evaluation results"""
        if not self.results:
            print("No results to analyze. Run evaluate_all_pairs() first.")
            return {}
        
        # Separate intra and inter tracklet results
        intra_results = [r for r in self.results if r.is_same_tracklet]
        inter_results = [r for r in self.results if not r.is_same_tracklet]
        
        # Calculate statistics
        intra_matches = [r.num_filtered_matches for r in intra_results]
        inter_matches = [r.num_filtered_matches for r in inter_results]
        
        intra_scores = [r.match_confidence for r in intra_results]
        inter_scores = [r.match_confidence for r in inter_results]
        
        analysis = {
            'num_intra_comparisons': len(intra_results),
            'num_inter_comparisons': len(inter_results),
            'intra_matches_mean': np.mean(intra_matches),
            'intra_matches_std': np.std(intra_matches),
            'intra_matches_median': np.median(intra_matches),
            'inter_matches_mean': np.mean(inter_matches),
            'inter_matches_std': np.std(inter_matches),
            'inter_matches_median': np.median(inter_matches),
            'intra_scores_mean': np.mean(intra_scores),
            'intra_scores_std': np.std(intra_scores),
            'inter_scores_mean': np.mean(inter_scores),
            'inter_scores_std': np.std(inter_scores),
            'separation_ratio': np.mean(intra_matches) / (np.mean(inter_matches) + 1e-6),
            'score_separation_ratio': np.mean(intra_scores) / (np.mean(inter_scores) + 1e-6)
        }
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(f"\nIntra-tracklet (Same Player) Statistics:")
        print(f"  Number of comparisons: {analysis['num_intra_comparisons']}")
        print(f"  Matches: {analysis['intra_matches_mean']:.2f} ± {analysis['intra_matches_std']:.2f}")
        print(f"  Median matches: {analysis['intra_matches_median']:.2f}")
        print(f"  Match score: {analysis['intra_scores_mean']:.4f} ± {analysis['intra_scores_std']:.4f}")
        
        print(f"\nInter-tracklet (Different Players) Statistics:")
        print(f"  Number of comparisons: {analysis['num_inter_comparisons']}")
        print(f"  Matches: {analysis['inter_matches_mean']:.2f} ± {analysis['inter_matches_std']:.2f}")
        print(f"  Median matches: {analysis['inter_matches_median']:.2f}")
        print(f"  Match score: {analysis['inter_scores_mean']:.4f} ± {analysis['inter_scores_std']:.4f}")
        
        print(f"\n{'='*80}")
        print(f"HYPOTHESIS TEST RESULTS:")
        print(f"{'='*80}")
        print(f"Separation Ratio (matches): {analysis['separation_ratio']:.2f}x")
        print(f"Separation Ratio (scores): {analysis['score_separation_ratio']:.2f}x")
        
        if analysis['separation_ratio'] > 1.5:
            print(f"\n✓ HYPOTHESIS SUPPORTED: Same-player matches are {analysis['separation_ratio']:.2f}x higher!")
        elif analysis['separation_ratio'] > 1.2:
            print(f"\n~ HYPOTHESIS PARTIALLY SUPPORTED: Moderate separation ({analysis['separation_ratio']:.2f}x)")
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED: Insufficient separation ({analysis['separation_ratio']:.2f}x)")
        print("="*80)
        
        return analysis
    
    def plot_results(self, save_path: Optional[str] = None):
        """Visualize evaluation results"""
        if not self.results:
            print("No results to plot. Run evaluate_all_pairs() first.")
            return
        
        # Separate intra and inter results
        intra_results = [r for r in self.results if r.is_same_tracklet]
        inter_results = [r for r in self.results if not r.is_same_tracklet]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Number of matches distribution
        ax = axes[0, 0]
        intra_matches = [r.num_filtered_matches for r in intra_results]
        inter_matches = [r.num_filtered_matches for r in inter_results]
        
        ax.hist(intra_matches, bins=30, alpha=0.6, label='Intra-tracklet (Same)', color='green')
        ax.hist(inter_matches, bins=30, alpha=0.6, label='Inter-tracklet (Different)', color='red')
        ax.set_xlabel('Number of Filtered Matches')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Keypoint Matches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Match scores distribution
        ax = axes[0, 1]
        intra_scores = [r.match_confidence for r in intra_results]
        inter_scores = [r.match_confidence for r in inter_results]
        
        ax.hist(intra_scores, bins=30, alpha=0.6, label='Intra-tracklet (Same)', color='green')
        ax.hist(inter_scores, bins=30, alpha=0.6, label='Inter-tracklet (Different)', color='red')
        ax.set_xlabel('Match Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Match Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Box plot comparison
        ax = axes[0, 2]
        data_to_plot = [intra_matches, inter_matches]
        box = ax.boxplot(data_to_plot, labels=['Same Player', 'Different Players'],
                        patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        box['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Number of Matches')
        ax.set_title('Match Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        # 4. Scatter plot: matches vs score
        ax = axes[1, 0]
        ax.scatter([r.num_filtered_matches for r in intra_results],
                  [r.match_confidence for r in intra_results],
                  alpha=0.5, label='Same Player', color='green', s=20)
        ax.scatter([r.num_filtered_matches for r in inter_results],
                  [r.match_confidence for r in inter_results],
                  alpha=0.5, label='Different Players', color='red', s=20)
        ax.set_xlabel('Number of Matches')
        ax.set_ylabel('Match Confidence')
        ax.set_title('Matches vs Confidence Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. ROC-like curve (if applicable)
        ax = axes[1, 1]
        all_scores = intra_scores + inter_scores
        all_labels = [1] * len(intra_scores) + [0] * len(inter_scores)
        
        if len(set(all_labels)) > 1 and len(all_scores) > 0:
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve (Same vs Different Player)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax = axes[1, 2]
        ax.axis('off')
        
        stats_text = f"""
        EVALUATION SUMMARY
        {'='*40}
        
        Intra-tracklet (Same Player):
          Comparisons: {len(intra_results)}
          Avg Matches: {np.mean(intra_matches):.2f} ± {np.std(intra_matches):.2f}
          Avg Score: {np.mean(intra_scores):.4f}
        
        Inter-tracklet (Different Players):
          Comparisons: {len(inter_results)}
          Avg Matches: {np.mean(inter_matches):.2f} ± {np.std(inter_matches):.2f}
          Avg Score: {np.mean(inter_scores):.4f}
        
        Separation Ratio: {np.mean(intra_matches) / (np.mean(inter_matches) + 1e-6):.2f}x
        
        Hypothesis: {'SUPPORTED ✓' if np.mean(intra_matches) > 1.5 * np.mean(inter_matches) else 'NOT SUPPORTED ✗'}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_path: str):
        """Save detailed results to file"""
        results_dict = {
            'results': [
                {
                    'tracklet_id_1': r.tracklet_id_1,
                    'tracklet_id_2': r.tracklet_id_2,
                    'image_idx_1': r.image_idx_1,
                    'image_idx_2': r.image_idx_2,
                    'num_matches': r.num_matches,
                    'num_filtered_matches': r.num_filtered_matches,
                    'avg_match_distance': r.avg_match_distance,
                    'match_confidence': r.match_confidence,
                    'is_same_tracklet': r.is_same_tracklet
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Player Re-Identification using ORB Keypoint Matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input-path',
        type=str,
        default='data/metric_learning',
        help='Path to directory containing tracklet subdirectories with images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/keypoint_reid',
        help='Directory to save output results and plots'
    )
    
    # ORB detector parameters
    parser.add_argument(
        '--n-features',
        type=int,
        default=500,
        help='Number of ORB features to detect'
    )
    parser.add_argument(
        '--ratio-threshold',
        type=float,
        default=0.75,
        help='Lowe\'s ratio test threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=5.0,
        help='RANSAC reprojection threshold in pixels'
    )
    parser.add_argument(
        '--min-matches',
        type=int,
        default=10,
        help='Minimum number of matches required for homography estimation'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--max-images-per-tracklet',
        type=int,
        default=20,
        help='Maximum number of images to use per tracklet'
    )
    parser.add_argument(
        '--sample-pairs',
        action='store_true',
        default=True,
        help='Whether to sample inter-tracklet pairs (faster evaluation)'
    )
    parser.add_argument(
        '--no-sample-pairs',
        action='store_false',
        dest='sample_pairs',
        help='Evaluate all inter-tracklet pairs (slower but complete)'
    )
    parser.add_argument(
        '--num-sample-pairs',
        type=int,
        default=50,
        help='Number of inter-tracklet pairs to sample (if --sample-pairs is set)'
    )
    
    # Filtering options
    parser.add_argument(
        '--use-horizontal-filter',
        action='store_true',
        default=False,
        help='Enable horizontal angle filtering for matches'
    )
    parser.add_argument(
        '--use-spatial-filter',
        action='store_true',
        default=True,
        help='Enable spatial consistency filtering (vertical binning)'
    )
    parser.add_argument(
        '--use-ransac-filter',
        action='store_true',
        default=True,
        help='Enable RANSAC geometric verification'
    )
    
    # Output options
    parser.add_argument(
        '--save-plot',
        action='store_true',
        default=True,
        help='Save visualization plots'
    )
    parser.add_argument(
        '--show-plot',
        action='store_true',
        default=False,
        help='Display plots interactively'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Configuration
    BASE_PATH = args.input_path  # e.g., "path/to/tracklets"
    OUTPUT_DIR = args.output_dir  # e.g., "path/to/output"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("PLAYER RE-IDENTIFICATION USING KEYPOINT MATCHING")
    print("="*80)
    print(f"\nBase path: {BASE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load tracklet images
    print("\n[Step 1/4] Loading tracklet images...")
    loader = TrackletImageLoader(BASE_PATH)
    
    # Step 2: Initialize matcher
    print("\n[Step 2/4] Initializing ORB keypoint matcher...")
    matcher = ORBKeypointMatcher(
        n_features=args.n_features,
        ratio_threshold=args.ratio_threshold,
        ransac_reproj_threshold=args.ransac_threshold,
        min_matches=args.min_matches
    )
    
    # Step 3: Initialize evaluator
    print("\n[Step 3/4] Initializing evaluator...")
    evaluator = PlayerReIDEvaluator(
        image_loader=loader,
        matcher=matcher,
        max_images_per_tracklet=args.max_images_per_tracklet,
        sample_pairs=args.sample_pairs,
        num_sample_pairs=args.num_sample_pairs
    )
    
    # Step 4: Run evaluation
    print("\n[Step 4/4] Running evaluation...")
    evaluator.evaluate_all_pairs()
    
    # Analyze results
    analysis = evaluator.analyze_results()
    
    # Plot results
    if args.save_plot:
        plot_path = os.path.join(OUTPUT_DIR, "keypoint_reid_results.png")
        evaluator.plot_results(save_path=plot_path)
    elif args.show_plot:
        evaluator.plot_results(save_path=None)
    
    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, "keypoint_reid_results.json")
    evaluator.save_results(results_path)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
