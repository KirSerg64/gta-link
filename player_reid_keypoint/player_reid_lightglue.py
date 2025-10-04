"""
Player Re-Identification using Deep Learning Keypoint Matching (LightGlue + SuperPoint)

This module implements a modern deep learning approach using:
- SuperPoint: Self-supervised keypoint detector
- LightGlue: Learned feature matcher with confidence scores

Advantages over classical methods:
- Better invariance to lighting, viewpoint, and pose changes
- Learned features specifically for matching
- Built-in confidence scores for matches

Requirements:
    pip install torch torchvision
    pip install git+https://github.com/cvg/LightGlue.git

Author: CV Algorithm Developer
Date: 2025-10-04
"""

import cv2 as cv
import numpy as np
import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

# Import LightGlue components
try:
    from lightglue import LightGlue, SuperPoint, DISK
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("WARNING: LightGlue not available. Install with: pip install git+https://github.com/cvg/LightGlue.git")


@dataclass
class MatchResult:
    """Store matching results between two images"""
    tracklet_id_1: int
    tracklet_id_2: int
    image_idx_1: int
    image_idx_2: int
    num_matches: int
    num_filtered_matches: int
    avg_match_confidence: float
    match_score: float
    is_same_tracklet: bool


class LightGlueKeypointMatcher:
    """Deep learning-based keypoint matching using SuperPoint + LightGlue"""
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_keypoints: int = 1024,
                 confidence_threshold: float = 0.5,
                 use_disk: bool = False):
        """
        Args:
            device: 'cuda' or 'cpu'
            max_keypoints: Maximum number of keypoints to extract
            confidence_threshold: Minimum confidence for valid matches
            use_disk: Use DISK detector instead of SuperPoint
        """
        if not LIGHTGLUE_AVAILABLE:
            raise ImportError("LightGlue is not installed. Please install it first.")
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        print(f"Initializing LightGlue matcher on device: {device}")
        
        # Initialize keypoint detector
        if use_disk:
            print("Using DISK keypoint detector")
            self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(device)
            self.matcher = LightGlue(features='disk').eval().to(device)
        else:
            print("Using SuperPoint keypoint detector")
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
            self.matcher = LightGlue(features='superpoint').eval().to(device)
    
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor format expected by LightGlue
        
        Args:
            img: BGR image (OpenCV format)
        
        Returns:
            Tensor of shape (1, 1, H, W) for grayscale
        """
        # Convert BGR to grayscale
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(gray)[None, None].to(self.device)
        
        return tensor
    
    @torch.no_grad()
    def extract_features(self, img: np.ndarray) -> Dict:
        """
        Extract keypoints and descriptors using SuperPoint/DISK
        
        Returns:
            Dictionary with 'keypoints', 'descriptors', 'keypoint_scores'
        """
        # Preprocess image
        img_tensor = self.preprocess_image(img)
        
        # Extract features
        features = self.extractor.extract(img_tensor)
        
        return features
    
    @torch.no_grad()
    def match_features(self, 
                       features1: Dict, 
                       features2: Dict,
                       img_shape1: Tuple[int, int],
                       img_shape2: Tuple[int, int]) -> Dict:
        """
        Match features between two images using LightGlue
        
        Args:
            features1: Features from image 1
            features2: Features from image 2
            img_shape1: (height, width) of image 1
            img_shape2: (height, width) of image 2
        
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
            'matches': filtered_matches,
            'confidence': filtered_confidence,
            'num_matches': len(filtered_matches),
            'avg_confidence': np.mean(filtered_confidence) if len(filtered_confidence) > 0 else 0.0
        }
    
    def filter_matches_spatial_consistency(self,
                                          kp1: np.ndarray,
                                          kp2: np.ndarray,
                                          matches: np.ndarray,
                                          confidence: np.ndarray,
                                          num_bins: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter matches based on spatial consistency (vertical binning)
        
        Args:
            kp1: Keypoints from image 1 (N, 2)
            kp2: Keypoints from image 2 (M, 2)
            matches: Match indices (K,) where matches[i] is the index in kp2 for kp1[i]
            confidence: Confidence scores for matches
            num_bins: Number of vertical bins
        
        Returns:
            Tuple of (filtered_matches, filtered_confidence)
        """
        if len(matches) == 0:
            return matches, confidence
        
        # Get matched keypoints
        matched_kp1 = kp1[np.arange(len(kp1))[:len(matches)]]
        matched_kp2 = kp2[matches]
        
        # Calculate vertical bins
        max_y1 = np.max(kp1[:, 1]) if len(kp1) > 0 else 1
        max_y2 = np.max(kp2[:, 1]) if len(kp2) > 0 else 1
        
        bin1 = (matched_kp1[:, 1] / max_y1 * num_bins).astype(int)
        bin2 = (matched_kp2[:, 1] / max_y2 * num_bins).astype(int)
        
        # Keep matches in same or adjacent bins
        valid_mask = np.abs(bin1 - bin2) <= 1
        
        filtered_matches = matches[valid_mask]
        filtered_confidence = confidence[valid_mask]
        
        return filtered_matches, filtered_confidence
    
    def match_images(self,
                    img1: np.ndarray,
                    img2: np.ndarray,
                    use_spatial_filter: bool = True) -> Dict:
        """
        Match two images and return detailed statistics
        
        Returns:
            Dictionary with matching statistics
        """
        # Extract features
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        
        # Get image shapes
        img_shape1 = img1.shape[:2]
        img_shape2 = img2.shape[:2]
        
        # Match features
        match_result = self.match_features(features1, features2, img_shape1, img_shape2)
        
        matches = match_result['matches']
        confidence = match_result['confidence']
        num_raw_matches = match_result['num_matches']
        
        # Extract keypoints as numpy arrays
        kp1 = features1['keypoints'][0].cpu().numpy()
        kp2 = features2['keypoints'][0].cpu().numpy()
        
        # Apply spatial filtering
        num_filtered_matches = num_raw_matches
        if use_spatial_filter and len(matches) > 0:
            # Get indices of matched keypoints
            match_indices = np.where(match_result['matches'] != -1)[0]
            if len(match_indices) > 0:
                matched_kp1 = kp1[match_indices]
                matched_kp2 = kp2[matches]
                
                # Simple vertical binning filter
                max_y1 = np.max(kp1[:, 1]) if len(kp1) > 0 else 1
                max_y2 = np.max(kp2[:, 1]) if len(kp2) > 0 else 1
                
                bin1 = (matched_kp1[:, 1] / max_y1 * 3).astype(int)
                bin2 = (matched_kp2[:, 1] / max_y2 * 3).astype(int)
                
                valid_mask = np.abs(bin1 - bin2) <= 1
                num_filtered_matches = np.sum(valid_mask)
                
                if np.sum(valid_mask) > 0:
                    confidence = confidence[valid_mask]
        
        # Calculate match score
        avg_confidence = np.mean(confidence) if len(confidence) > 0 else 0.0
        match_score = num_filtered_matches / max(len(kp1), len(kp2), 1)
        
        return {
            'num_kp1': len(kp1),
            'num_kp2': len(kp2),
            'num_raw_matches': num_raw_matches,
            'num_filtered_matches': num_filtered_matches,
            'avg_match_confidence': avg_confidence,
            'match_score': match_score
        }


class LightGluePlayerReIDEvaluator:
    """Evaluate player re-identification using LightGlue"""
    
    def __init__(self,
                 base_path: str,
                 matcher: LightGlueKeypointMatcher,
                 max_images_per_tracklet: int = 20,
                 sample_pairs: bool = True,
                 num_sample_pairs: int = 100):
        """
        Args:
            base_path: Path to directory containing tracklet subdirectories
            matcher: LightGlue matcher instance
            max_images_per_tracklet: Maximum images to use per tracklet
            sample_pairs: Whether to sample image pairs
            num_sample_pairs: Number of pairs to sample for inter-tracklet comparison
        """
        self.base_path = Path(base_path)
        self.matcher = matcher
        self.max_images_per_tracklet = max_images_per_tracklet
        self.sample_pairs = sample_pairs
        self.num_sample_pairs = num_sample_pairs
        
        self.tracklets = {}
        self.results: List[MatchResult] = []
        
        self._load_tracklets()
    
    def _load_tracklets(self):
        """Load tracklet images from directory"""
        for subdir in sorted(self.base_path.iterdir()):
            if subdir.is_dir() and subdir.name.isdigit():
                tracklet_id = int(subdir.name)
                image_paths = sorted(list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")))
                
                if len(image_paths) > 0:
                    self.tracklets[tracklet_id] = [str(p) for p in image_paths]
        
        print(f"Loaded {len(self.tracklets)} tracklets")
        for tid, paths in self.tracklets.items():
            print(f"  Tracklet {tid}: {len(paths)} images")
    
    def evaluate_all_pairs(self):
        """Run complete evaluation"""
        tracklet_ids = sorted(self.tracklets.keys())
        
        print("\n" + "="*80)
        print("LIGHTGLUE PLAYER RE-IDENTIFICATION EVALUATION")
        print("="*80)
        
        self.results = []
        
        # Intra-tracklet evaluation
        print("\n[1/2] Evaluating INTRA-TRACKLET matches...")
        self._evaluate_intra_tracklet(tracklet_ids)
        
        # Inter-tracklet evaluation
        print("\n[2/2] Evaluating INTER-TRACKLET matches...")
        self._evaluate_inter_tracklet(tracklet_ids)
        
        print(f"\nTotal evaluations: {len(self.results)}")
    
    def _evaluate_intra_tracklet(self, tracklet_ids: List[int]):
        """Evaluate same-player matches"""
        for tid in tqdm(tracklet_ids, desc="Intra-tracklet"):
            image_paths = self.tracklets[tid]
            
            # Sample images
            if len(image_paths) > self.max_images_per_tracklet:
                indices = np.linspace(0, len(image_paths)-1, 
                                    self.max_images_per_tracklet, dtype=int)
                image_paths = [image_paths[i] for i in indices]
            
            # Load images
            images = [cv.imread(p) for p in image_paths]
            
            # Match all pairs
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    stats = self.matcher.match_images(images[i], images[j])
                    
                    result = MatchResult(
                        tracklet_id_1=tid,
                        tracklet_id_2=tid,
                        image_idx_1=i,
                        image_idx_2=j,
                        num_matches=stats['num_raw_matches'],
                        num_filtered_matches=stats['num_filtered_matches'],
                        avg_match_confidence=stats['avg_match_confidence'],
                        match_score=stats['match_score'],
                        is_same_tracklet=True
                    )
                    self.results.append(result)
    
    def _evaluate_inter_tracklet(self, tracklet_ids: List[int]):
        """Evaluate different-player matches"""
        # Generate tracklet pairs
        pairs = [(tid1, tid2) for i, tid1 in enumerate(tracklet_ids)
                for tid2 in tracklet_ids[i+1:]]
        
        if self.sample_pairs and len(pairs) > self.num_sample_pairs:
            pairs = [pairs[i] for i in np.random.choice(
                len(pairs), self.num_sample_pairs, replace=False)]
        
        # Load all images
        tracklet_images = {}
        for tid in tracklet_ids:
            paths = self.tracklets[tid]
            if len(paths) > self.max_images_per_tracklet:
                indices = np.linspace(0, len(paths)-1, 
                                    self.max_images_per_tracklet, dtype=int)
                paths = [paths[i] for i in indices]
            tracklet_images[tid] = [cv.imread(p) for p in paths]
        
        # Match pairs
        for tid1, tid2 in tqdm(pairs, desc="Inter-tracklet"):
            images1 = tracklet_images[tid1]
            images2 = tracklet_images[tid2]
            
            # Sample a few image pairs
            num_comps = min(5, len(images1) * len(images2))
            for _ in range(num_comps):
                i = np.random.randint(0, len(images1))
                j = np.random.randint(0, len(images2))
                
                stats = self.matcher.match_images(images1[i], images2[j])
                
                result = MatchResult(
                    tracklet_id_1=tid1,
                    tracklet_id_2=tid2,
                    image_idx_1=i,
                    image_idx_2=j,
                    num_matches=stats['num_raw_matches'],
                    num_filtered_matches=stats['num_filtered_matches'],
                    avg_match_confidence=stats['avg_match_confidence'],
                    match_score=stats['match_score'],
                    is_same_tracklet=False
                )
                self.results.append(result)
    
    def analyze_results(self) -> Dict:
        """Analyze and print results"""
        intra = [r for r in self.results if r.is_same_tracklet]
        inter = [r for r in self.results if not r.is_same_tracklet]
        
        intra_matches = [r.num_filtered_matches for r in intra]
        inter_matches = [r.num_filtered_matches for r in inter]
        
        intra_conf = [r.avg_match_confidence for r in intra]
        inter_conf = [r.avg_match_confidence for r in inter]
        
        analysis = {
            'intra_matches_mean': np.mean(intra_matches),
            'intra_matches_std': np.std(intra_matches),
            'inter_matches_mean': np.mean(inter_matches),
            'inter_matches_std': np.std(inter_matches),
            'intra_conf_mean': np.mean(intra_conf),
            'inter_conf_mean': np.mean(inter_conf),
            'separation_ratio': np.mean(intra_matches) / (np.mean(inter_matches) + 1e-6)
        }
        
        print("\n" + "="*80)
        print("LIGHTGLUE EVALUATION RESULTS")
        print("="*80)
        print(f"\nIntra-tracklet: {analysis['intra_matches_mean']:.2f} ± {analysis['intra_matches_std']:.2f} matches")
        print(f"  Avg confidence: {analysis['intra_conf_mean']:.4f}")
        print(f"\nInter-tracklet: {analysis['inter_matches_mean']:.2f} ± {analysis['inter_matches_std']:.2f} matches")
        print(f"  Avg confidence: {analysis['inter_conf_mean']:.4f}")
        print(f"\nSeparation Ratio: {analysis['separation_ratio']:.2f}x")
        print("="*80)
        
        return analysis
    
    def save_results(self, output_path: str):
        """Save results to JSON"""
        results_dict = {
            'results': [
                {
                    'tracklet_id_1': r.tracklet_id_1,
                    'tracklet_id_2': r.tracklet_id_2,
                    'image_idx_1': r.image_idx_1,
                    'image_idx_2': r.image_idx_2,
                    'num_matches': r.num_matches,
                    'num_filtered_matches': r.num_filtered_matches,
                    'avg_match_confidence': r.avg_match_confidence,
                    'match_score': r.match_score,
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
        description='Player Re-Identification using LightGlue + SuperPoint',
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
        default='outputs/lightglue_reid',
        help='Directory to save output results'
    )
    
    # Device and model parameters
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for computation (auto will use cuda if available)'
    )
    parser.add_argument(
        '--max-keypoints',
        type=int,
        default=1024,
        help='Maximum number of keypoints to extract'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Minimum confidence threshold for valid matches (0.0-1.0)'
    )
    parser.add_argument(
        '--use-disk',
        action='store_true',
        default=False,
        help='Use DISK detector instead of SuperPoint'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--max-images-per-tracklet',
        type=int,
        default=15,
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
        '--use-spatial-filter',
        action='store_true',
        default=True,
        help='Enable spatial consistency filtering (vertical binning)'
    )
    
    return parser.parse_args()


def main():
    """Main execution"""
    # Parse arguments
    args = parse_arguments()
    
    BASE_PATH = args.input_path
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("LIGHTGLUE PLAYER RE-IDENTIFICATION")
    print("="*80)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize matcher
    matcher = LightGlueKeypointMatcher(
        device=device,
        max_keypoints=args.max_keypoints,
        confidence_threshold=args.confidence_threshold,
        use_disk=args.use_disk
    )
    
    # Initialize evaluator
    evaluator = LightGluePlayerReIDEvaluator(
        base_path=BASE_PATH,
        matcher=matcher,
        max_images_per_tracklet=args.max_images_per_tracklet,
        sample_pairs=args.sample_pairs,
        num_sample_pairs=args.num_sample_pairs
    )
    
    # Run evaluation
    evaluator.evaluate_all_pairs()
    
    # Analyze
    analysis = evaluator.analyze_results()
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "lightglue_results.json")
    evaluator.save_results(results_path)
    
    print(f"\n✓ Complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
