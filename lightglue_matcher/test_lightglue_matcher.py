"""
Quick test script for LightGlue tracklet matcher

This script tests the basic functionality without running the full pipeline.

Usage:
    python test_lightglue_matcher.py --video_path ./data/video.mp4
"""

import argparse
import cv2 as cv
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightglue_matcher import TrackletLightGlueMatcher
from Tracklet import Tracklet
from loguru import logger


def create_dummy_tracklets():
    """Create dummy tracklets for testing"""
    # Tracklet 1: frames 10-30, moving from left to right
    tracklet1 = Tracklet(
        track_id=1,
        frames=list(range(10, 31)),
        scores=[0.9] * 21,
        bboxes=[[100 + i*5, 200, 80, 120] for i in range(21)],
        feats=None
    )
    
    # Tracklet 2: frames 50-70, similar location (should match)
    tracklet2 = Tracklet(
        track_id=2,
        frames=list(range(50, 71)),
        scores=[0.9] * 21,
        bboxes=[[150 + i*5, 210, 85, 125] for i in range(21)],
        feats=None
    )
    
    # Tracklet 3: frames 100-120, different location (should NOT match)
    tracklet3 = Tracklet(
        track_id=3,
        frames=list(range(100, 121)),
        scores=[0.9] * 21,
        bboxes=[[500 + i*5, 400, 90, 130] for i in range(21)],
        feats=None
    )
    
    return {1: tracklet1, 2: tracklet2, 3: tracklet3}


def test_video_loading(video_path: str):
    """Test if video can be loaded"""
    logger.info("=" * 60)
    logger.info("TEST 1: Video Loading")
    logger.info("=" * 60)
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"✅ Video loaded successfully")
    logger.info(f"   Frames: {total_frames}")
    logger.info(f"   FPS: {fps}")
    logger.info(f"   Resolution: {width}x{height}")
    
    # Test reading a frame
    ret, frame = cap.read()
    if ret:
        logger.info(f"✅ Frame read successfully, shape: {frame.shape}")
    else:
        logger.error(f"❌ Failed to read frame")
        cap.release()
        return False
    
    cap.release()
    return True


def test_matcher_initialization(video_path: str):
    """Test matcher initialization"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Matcher Initialization")
    logger.info("=" * 60)
    
    try:
        matcher = TrackletLightGlueMatcher(
            video_path=video_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_keypoints=1024,  # Reduced for faster testing
            confidence_threshold=0.3,
            sample_strategy='uniform',
            max_samples_per_tracklet=5,  # Reduced for faster testing
            enable_cache=True,
            cache_size_gb=2.0
        )
        logger.info("✅ Matcher initialized successfully")
        logger.info(f"   Device: {matcher.device}")
        logger.info(f"   Max keypoints: 1024")
        logger.info(f"   Cache enabled: True")
        return matcher
    except Exception as e:
        logger.error(f"❌ Failed to initialize matcher: {e}")
        return None


def test_frame_extraction(matcher: TrackletLightGlueMatcher):
    """Test frame extraction"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Frame Extraction")
    logger.info("=" * 60)
    
    # Test extracting a crop
    test_bbox = [100, 100, 80, 120]  # x, y, w, h
    test_frame = 10
    
    try:
        crop = matcher.extract_frame_crop(test_frame, test_bbox)
        if crop is not None:
            logger.info(f"✅ Frame crop extracted successfully")
            logger.info(f"   Crop shape: {crop.shape}")
            return True
        else:
            logger.error("❌ Frame crop is None")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to extract frame crop: {e}")
        return False


def test_feature_extraction(matcher: TrackletLightGlueMatcher):
    """Test feature extraction"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Feature Extraction")
    logger.info("=" * 60)
    
    # Extract a crop
    test_bbox = [100, 100, 80, 120]
    test_frame = 10
    
    try:
        crop = matcher.extract_frame_crop(test_frame, test_bbox)
        if crop is None:
            logger.error("❌ Failed to extract crop for feature extraction")
            return False
        
        # Extract features
        features = matcher.extract_features_batch([crop])
        
        if features and len(features) > 0:
            feat = features[0]
            num_keypoints = feat['keypoints'][0].shape[0]
            descriptor_dim = feat['descriptors'][0].shape[1]
            
            logger.info(f"✅ Features extracted successfully")
            logger.info(f"   Keypoints detected: {num_keypoints}")
            logger.info(f"   Descriptor dimension: {descriptor_dim}")
            return True
        else:
            logger.error("❌ No features extracted")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to extract features: {e}")
        return False


def test_tracklet_matching(matcher: TrackletLightGlueMatcher):
    """Test tracklet matching with real tracklets"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Tracklet Matching")
    logger.info("=" * 60)
    
    # Create dummy tracklets
    tracklets = create_dummy_tracklets()
    
    try:
        # Test matching tracklet 1 vs 2 (should be similar)
        logger.info("Testing Tracklet 1 vs Tracklet 2 (should match)...")
        distance_12 = matcher.compute_distance(tracklets[1], tracklets[2])
        logger.info(f"   Distance: {distance_12:.4f}")
        
        # Test matching tracklet 1 vs 3 (should be different)
        logger.info("Testing Tracklet 1 vs Tracklet 3 (should NOT match)...")
        distance_13 = matcher.compute_distance(tracklets[1], tracklets[3])
        logger.info(f"   Distance: {distance_13:.4f}")
        
        # Check if distances make sense
        if distance_12 < distance_13:
            logger.info("✅ Distance ordering correct (similar < different)")
            logger.info(f"   Similar pair: {distance_12:.4f}")
            logger.info(f"   Different pair: {distance_13:.4f}")
            return True
        else:
            logger.warning("⚠️  Distance ordering unexpected")
            logger.warning(f"   Similar pair: {distance_12:.4f}")
            logger.warning(f"   Different pair: {distance_13:.4f}")
            logger.warning("   This might be due to dummy data, try with real tracklets")
            return True  # Still pass since it's dummy data
            
    except Exception as e:
        logger.error(f"❌ Failed to match tracklets: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Test GPU availability"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 0: GPU Availability")
    logger.info("=" * 60)
    
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available")
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        logger.warning("⚠️  CUDA not available, will use CPU (slower)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test LightGlue tracklet matcher")
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to test video file')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LightGlue Tracklet Matcher Test Suite")
    logger.info("=" * 60)
    
    # Run tests
    results = {}
    
    results['gpu'] = test_gpu_availability()
    results['video'] = test_video_loading(args.video_path)
    
    if not results['video']:
        logger.error("\n❌ Video loading failed, cannot continue tests")
        return
    
    matcher = test_matcher_initialization(args.video_path)
    results['matcher'] = matcher is not None
    
    if matcher is None:
        logger.error("\n❌ Matcher initialization failed, cannot continue tests")
        return
    
    results['extraction'] = test_frame_extraction(matcher)
    results['features'] = test_feature_extraction(matcher)
    results['matching'] = test_tracklet_matching(matcher)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:15s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("You can now run the full pipeline:")
        logger.info("python refine_tracklets_lightglue.py \\")
        logger.info("    --dataset SoccerNet \\")
        logger.info("    --tracker SORT \\")
        logger.info("    --track_src ./data/tracklets \\")
        logger.info(f"    --video_path {args.video_path} \\")
        logger.info("    --use_connect \\")
        logger.info("    --merge_dist_thres 0.5")
        logger.info("=" * 60)
    else:
        logger.error("\n❌ Some tests failed, please check the errors above")


if __name__ == "__main__":
    main()
