"""
Example: Using LightGlue Matcher as a Module

This script demonstrates how to use the lightglue_matcher module
for tracklet distance calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightglue_matcher import TrackletLightGlueMatcher
from Tracklet import Tracklet
import numpy as np

def example_basic_usage():
    """Example 1: Basic usage"""
    print("=" * 60)
    print("Example 1: Basic Distance Calculation")
    print("=" * 60)
    
    # Initialize matcher
    matcher = TrackletLightGlueMatcher(
        video_path="../data/7_06_25fps_2min.mp4",  # Adjust path
        device='cuda',
        max_keypoints=1024,
        max_samples_per_tracklet=5
    )
    
    # Create sample tracklets
    tracklet1 = Tracklet(
        track_id=1,
        frames=[10, 11, 12, 13, 14],
        scores=[0.9] * 5,
        bboxes=[[100, 200, 80, 120]] * 5
    )
    
    tracklet2 = Tracklet(
        track_id=2,
        frames=[50, 51, 52, 53, 54],
        scores=[0.9] * 5,
        bboxes=[[110, 210, 85, 125]] * 5
    )
    
    # Compute distance
    distance = matcher.compute_distance(tracklet1, tracklet2)
    print(f"Distance between tracklets: {distance:.4f}")
    print(f"Interpretation: {'Same player' if distance < 0.5 else 'Different players'}")


def example_distance_matrix():
    """Example 2: Compute full distance matrix"""
    print("\n" + "=" * 60)
    print("Example 2: Distance Matrix Computation")
    print("=" * 60)
    
    # Initialize matcher
    matcher = TrackletLightGlueMatcher(
        video_path="../data/7_06_25fps_2min.mp4",
        device='cuda',
        max_keypoints=1024,
        max_samples_per_tracklet=3
    )
    
    # Create multiple tracklets
    tracklets = {}
    for i in range(3):
        tracklets[i] = Tracklet(
            track_id=i,
            frames=list(range(i*50, i*50 + 10)),
            scores=[0.9] * 10,
            bboxes=[[100 + i*20, 200, 80, 120]] * 10
        )
    
    # Compute distance matrix
    print("Computing distance matrix...")
    dist_matrix = matcher.compute_distance_matrix(tracklets)
    
    print("\nDistance Matrix:")
    print(dist_matrix)


def example_custom_configuration():
    """Example 3: Custom configuration"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Initialize with custom settings
    matcher = TrackletLightGlueMatcher(
        video_path="../data/7_06_25fps_2min.mp4",
        device='cuda',
        max_keypoints=2048,              # High quality
        confidence_threshold=0.2,         # Lower threshold
        sample_strategy='adaptive',       # Select best frames
        max_samples_per_tracklet=15,     # More samples
        use_clahe=True,                  # Enhance contrast
        enable_cache=True,               # Enable caching
        cache_size_gb=5.0
    )
    
    print("Matcher configured with:")
    print(f"  - Max keypoints: 2048")
    print(f"  - Sampling: adaptive")
    print(f"  - Samples per tracklet: 15")
    print(f"  - CLAHE: enabled")
    print(f"  - Cache: 5GB")


if __name__ == "__main__":
    print("LightGlue Matcher - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        # example_distance_matrix()  # Uncomment if you have real data
        example_custom_configuration()
        
        print("\n" + "=" * 60)
        print("✅ Examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Adjust video_path to your actual video")
        print("2. Load real tracklets from pkl files")
        print("3. Run full pipeline with refine_tracklets_lightglue.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Video file exists at specified path")
        print("2. LightGlue is installed: pip install git+https://github.com/cvg/LightGlue.git")
        print("3. CUDA is available (or use device='cpu')")
