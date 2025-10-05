# Video Visualization Feature

## Overview

The LightGlue tracklet refinement pipeline now supports automatic creation of visualization videos showing the refined tracklets with player IDs drawn on each detection.

## Usage

### Basic Video Creation

Add the `--create_video` flag to any refinement command:

```bash
python refine_tracklets_lightglue.py \
    --dataset SoccerNet \
    --tracker SORT \
    --track_src ./data/tracklets \
    --video_path ./data/7_06_25fps_2min.mp4 \
    --use_connect \
    --merge_dist_thres 0.5 \
    --create_video
```

### Custom Output Directory

Specify where to save the visualization videos:

```bash
python refine_tracklets_lightglue.py \
    --dataset SoccerNet \
    --tracker SORT \
    --track_src ./data/tracklets \
    --video_path ./data/7_06_25fps_2min.mp4 \
    --use_connect \
    --merge_dist_thres 0.5 \
    --create_video \
    --video_output_dir ./output/videos
```

## Output

### Video Naming

The output video will be named: `{sequence_name}_refined_lightglue.mp4`

For example, if processing `7_06_25fps_2min.pkl`, the output will be:
- `7_06_25fps_2min_refined_lightglue.mp4`

### Default Location

If `--video_output_dir` is not specified, videos are saved in the same directory as the tracking results:

```
data/
└── SORT_SoccerNet_Connect_LightGlue_kp2048_samples10_mergeDist0.5/
    ├── 7_06_25fps_2min.txt                        # Tracking results
    └── 7_06_25fps_2min_refined_lightglue.mp4      # Visualization video
```

### Custom Location

If `--video_output_dir` is specified:

```
output/
└── videos/
    └── 7_06_25fps_2min_refined_lightglue.mp4
```

## Visualization Details

The visualization includes:
- **Bounding boxes** around each detected player
- **Track IDs** displayed on each player (final refined IDs)
- **Unique colors** per track ID for easy visual tracking
- **Frame information** (optional, depending on drawer implementation)

## Technical Details

### Implementation

Uses the `create_final_tracklet_video()` function from `utils/video_creator.py`:

```python
from utils.video_creator import create_final_tracklet_video

create_final_tracklet_video(
    video_path=args.video_path,
    final_tracklets=mergedTracklets,
    output_path=video_output_path,
    show_trajectories=False
)
```

### Parameters

- **video_path**: Original video file
- **final_tracklets**: Dictionary of refined tracklets (after LightGlue matching and merging)
- **output_path**: Where to save the visualization video
- **show_trajectories**: Whether to draw trajectory trails (set to `False` by default)

### Video Format

- **Codec**: MP4V (MPEG-4)
- **FPS**: Same as input video
- **Resolution**: Same as input video
- **Size**: Typically 1-3x input video size (depends on compression)

## Use Cases

### 1. Visual Quality Inspection

Quickly verify that tracklet refinement is working correctly:

```bash
python refine_tracklets_lightglue.py \
    --dataset SoccerNet \
    --tracker SORT \
    --track_src ./data/tracklets \
    --video_path ./data/video.mp4 \
    --use_connect \
    --merge_dist_thres 0.5 \
    --create_video
```

Watch the output video to check:
- Are track IDs consistent across frames?
- Are false merges happening?
- Are tracklets properly split?

### 2. Parameter Tuning

Create videos with different parameters to compare:

```bash
# Strict merging
python refine_tracklets_lightglue.py \
    --track_src ./data/tracklets \
    --video_path ./data/video.mp4 \
    --use_connect \
    --merge_dist_thres 0.3 \
    --create_video \
    --video_output_dir ./output/strict

# Lenient merging
python refine_tracklets_lightglue.py \
    --track_src ./data/tracklets \
    --video_path ./data/video.mp4 \
    --use_connect \
    --merge_dist_thres 0.7 \
    --create_video \
    --video_output_dir ./output/lenient
```

Compare the two videos to find optimal `merge_dist_thres`.

### 3. Demo/Presentation

Create polished videos for presentations or papers:

```bash
python refine_tracklets_lightglue.py \
    --dataset SoccerNet \
    --tracker SORT \
    --track_src ./data/tracklets \
    --video_path ./data/video.mp4 \
    --use_connect \
    --merge_dist_thres 0.5 \
    --lightglue_max_keypoints 2048 \
    --lightglue_samples 15 \
    --create_video \
    --video_output_dir ./demo
```

## Performance Considerations

### Processing Time

Video creation adds approximately:
- **2-5 seconds per minute of video** (on typical hardware)
- Depends on video resolution and frame rate

### Storage

Output videos typically require:
- **1-3x the size of input video** (with MP4V compression)
- For 1080p video at 25fps: ~50-150 MB per minute

### Optimization

To minimize overhead:
- Video creation runs **after** all tracklet processing is complete
- Uses efficient OpenCV video writer
- Processes frames sequentially (low memory footprint)

## Error Handling

If video creation fails, the pipeline will:
1. Log the error
2. Continue with next sequences (if any)
3. Save tracklet results (they are not affected)

Example error message:
```
ERROR: Failed to create visualization video: Cannot open video file
```

Common issues:
- **Video file not found**: Check `--video_path`
- **Codec not available**: Install OpenCV with full codec support
- **Disk space**: Ensure sufficient space for output video
- **Permissions**: Check write permissions for output directory

## Advanced: Custom Visualization

If you need custom visualization (e.g., with trajectories, different colors), you can:

1. **Modify the call in `refine_tracklets_lightglue.py`:**

```python
create_final_tracklet_video(
    video_path=args.video_path,
    final_tracklets=mergedTracklets,
    output_path=video_output_path,
    show_trajectories=True  # Enable trajectory trails
)
```

2. **Use the standalone video creator:**

```python
from utils.video_creator import create_final_tracklet_video

create_final_tracklet_video(
    video_path="./data/video.mp4",
    final_tracklets=my_tracklets,
    output_path="./custom_output.mp4",
    show_trajectories=True
)
```

3. **Explore other video creator functions:**

See `utils/video_creator.py` for additional visualization options:
- `create_comparison_video()` - Side-by-side comparison
- `create_overlay_video()` - Overlay multiple tracklet sets

## Example Workflow

Complete workflow with visualization:

```bash
# 1. Extract tracklets (if not done)
python generate_tracklets.py \
    --video_path ./data/video.mp4 \
    --output ./data/tracklets

# 2. Refine with LightGlue + create video
python refine_tracklets_lightglue.py \
    --dataset SoccerNet \
    --tracker SORT \
    --track_src ./data/tracklets \
    --video_path ./data/video.mp4 \
    --use_connect \
    --merge_dist_thres 0.5 \
    --lightglue_max_keypoints 2048 \
    --lightglue_samples 10 \
    --create_video \
    --video_output_dir ./output/videos

# 3. View the result
# Open: output/videos/video_refined_lightglue.mp4
```

## Conclusion

The video visualization feature provides an easy way to:
- ✅ **Validate** tracklet refinement quality
- ✅ **Tune** parameters visually
- ✅ **Present** results in papers/demos
- ✅ **Debug** tracking issues

No additional tools or scripts needed - just add `--create_video`! 🎥
