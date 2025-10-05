"""
LightGlue-based Tracklet Matching Module

This module provides image-based tracklet distance calculation using
SuperPoint + LightGlue for improved player re-identification in sports videos.

Main Components:
- TrackletLightGlueMatcher: Core matching class
- refine_tracklets_lightglue.py: Modified pipeline script
- test_lightglue_matcher.py: Test suite
- compare_approaches.py: Comparison tool

Usage:
    from lightglue_matcher import TrackletLightGlueMatcher
    
    matcher = TrackletLightGlueMatcher(video_path="video.mp4")
    distance = matcher.compute_distance(tracklet1, tracklet2)

For detailed documentation, see README.md in this directory.
"""

from .tracklet_lightglue_matcher import (
    TrackletLightGlueMatcher,
    FrameCache,
    MatchStatistics,
    get_distance_lightglue
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__all__ = [
    'TrackletLightGlueMatcher',
    'FrameCache',
    'MatchStatistics',
    'get_distance_lightglue'
]
