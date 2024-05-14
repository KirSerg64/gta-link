import numpy as np
import os
import torch
import pickle
import sys
sys.path.append('..')
# print(sys.path)
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from loguru import logger
# from tracker.Deep_EIoU import STrack
import shutil # used for copying files
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering

class Tracklet_archive:
    def __init__(self):
        '''
        Initialize the Tracklet with None value fields
        '''
        self.track_id = None
        self.parent_id = None
        self.scores = None
        self.times = None
        self.bboxes = None
        self.features = None

    def __init__(self, track_id, frames, scores, bboxes, feats=None):
        '''
        Initialize the Tracklet with IDs, times, scores, bounding boxes, and optional features.
        - frames, scores can be lists or single elements.
        - bboxes can be a single list of 4 elements or a list of lists where each sublist has 4 elements.
        - feats should be a list of numpy arrays each of shape (512,) or None.
        '''
        self.track_id = track_id
        self.parent_id = None
        # Ensure inputs are list type; convert single elements to lists
        self.scores = scores if isinstance(scores, list) else [scores]
        self.times = frames if isinstance(frames, list) else [frames]

        self.bboxes = bboxes if (isinstance(bboxes[0], list)) else [bboxes]
        self.features = feats if feats is not None else []

    def append_det(self, frame, score, bbox):
        # frame (float), score (float), bbox (list(4))
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)

    def append_feat(self, feat):
        # feat (numpy array)
        self.features.append(feat)

    def extract(self, start, end):
        subtrack = Tracklet()
        subtrack.times = self.times[start : end + 1]
        subtrack.bboxes = self.bboxes[start : end + 1]
        subtrack.track_id = self.track_id
        subtrack.parent_tid = self.track_id
        return subtrack

class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None, feats=None):
        '''
        Initialize the Tracklet with IDs, times, scores, bounding boxes, and optional features.
        If parameters are not provided, initializes them to None or empty lists.

        Args:
            track_id (int, optional): Unique identifier for the track. Defaults to None.
            frames (list or int, optional): Frame numbers where the track is present. Can be a list of frames or a single frame. Defaults to None.
            scores (list or float, optional): Detection scores corresponding to frames. Can be a list of scores or a single score. Defaults to None.
            bboxes (list of lists or list, optional): Bounding boxes corresponding to each frame. Each bounding box is a list of 4 elements. Defaults to None.
            feats (list of np.array, optional): Feature vectors corresponding to frames. Each feature should be a numpy array of shape (512,). Defaults to None.
        '''
        self.track_id = track_id
        self.parent_id = None
        self.scores = scores if isinstance(scores, list) else [scores] if scores is not None else []
        self.times = frames if isinstance(frames, list) else [frames] if frames is not None else []
        self.bboxes = bboxes if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else [bboxes] if bboxes is not None else []
        self.features = feats if feats is not None else []

    def append_det(self, frame, score, bbox):
        '''
        Appends a detection to the tracklet.

        Args:
            frame (int): Frame number for the detection.
            score (float): Detection score.
            bbox (list of float): Bounding box with four elements [x, y, width, height].
        '''
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)

    def append_feat(self, feat):
        '''
        Appends a feature vector to the tracklet.

        Args:
            feat (np.array): Feature vector of shape (512,).
        '''
        self.features.append(feat)

    def extract(self, start, end):
        '''
        Extracts a subtrack from the tracklet between two indices.

        Args:
            start (int): Start index for the extraction.
            end (int): End index for the extraction.

        Returns:
            Tracklet: A new Tracklet object that is a subset of the original from start to end indices.
        '''
        subtrack = Tracklet(self.track_id, self.times[start:end + 1], self.scores[start:end + 1], self.bboxes[start:end + 1], self.features[start:end + 1] if self.features else None)
        return subtrack

# TODO:
# 1. Add comments to functions and hyperparameters
# 2. Test code
# 3. Delete unused lines/functions

PROCESS = 'Split+Connect'
# Define hyperparameters for merging tracklets
SPATIAL_FACTOR = 1        # spatial constraint factor restricting spatial distance between two targets to be merged
MERGE_DIST_THRES = 0.4            # Define the merging distance threshold (upper bound)

# DEFINE hyperparameters for splitting tracklets
LEN_THRES = 100
MAX_K = 5   # NOTE: higher number of clusters significantly increases runtime in distance map calculation
INNER_DIST_THRES = 0.4

'''def heatmap_archive(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    axls
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar'''

'''def annotate_heatmap_archive(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts'''

'''def displayDist_archive(Dist_dict, num_of_displays=5):
    seqs = sorted(Dist_dict.keys())
    if num_of_displays > len(seqs):
        num_of_displays = len(seqs)
    for i in range(num_of_displays):
        seq = seqs[i]
        Dist = Dist_dict[seq]
        # fig, ax = plt.subplots()
        # ticks = np.arange(len(Dist))
        # im, cbar = heatmap_archive(Dist, ticks, ticks, ax=ax, cmap='binary', cbarlabel='distance')
        # texts = annotate_heatmap_archive(im, valfmt="{x:.2f}")
        # fig.tight_layout()
        # plt.show()

        plt.imshow(Dist, cmap='binary')
        plt.colorbar()
        plt.title(seq)
        plt.show()'''

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
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2
        '''Optionally eliminate false positive subtracks
        if (s1_end - s1_start + 1) < 30:
            seg1.pop(0)  # Remove the first element from seg1
            continue
        if (s2_end - s2_start + 1) < 30:
            seg2.pop(0)  # Remove the first element from seg2
            continue
        '''

        # subtrack_1 = get_subtrack(track1, s1_start, s1_end)  # Extract subtrack from track 1
        # subtrack_2 = get_subtrack(track2, s2_start, s2_end)  # Extract subtrack from track 2
        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.times[s2_start]  # Get the starting frame of subtrack 2

        # print("track 1 and 2 start frame:", s1_startFrame, s2_startFrame)
        # print("track 1 and 2 end frame:", track1.times[s1_end], track2.times[s2_end])

        if s1_startFrame < s2_startFrame:  # Compare the starting frames of the two subtracks
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
        # subtracks.append(get_subtrack(track_remain, s_start, s_end))
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks  # Return the list of valid subtracks sorted ascending temporally

def query_subtracks_new(seg1, seg2, track1, track2):
    """
    Modifies the previous function to handle tracks that may not strictly follow one another
    but still need to be considered for merging or comparison.
    """
    subtracks = []
    index1, index2 = 0, 0

    while index1 < len(seg1) and index2 < len(seg2):
        s1_start, s1_end = seg1[index1]
        s2_start, s2_end = seg2[index2]
        s1_startFrame = track1.times[s1_start]
        s1_endFrame = track1.times[s1_end]
        s2_startFrame = track2.times[s2_start]
        s2_endFrame = track2.times[s2_end]

        # Flexible handling based on start and end times, allowing for close proximity
        if s1_endFrame < s2_startFrame:
            subtracks.append(get_subtrack(track1, s1_start, s1_end))
            index1 += 1
        elif s2_endFrame < s1_startFrame:
            subtracks.append(get_subtrack(track2, s2_start, s2_end))
            index2 += 1
        else:
            # Handle overlap or close proximity
            subtracks.append(get_subtrack(track1, s1_start, s1_end))
            subtracks.append(get_subtrack(track2, s2_start, s2_end))
            index1 += 1
            index2 += 1

    # Add remaining segments if any
    while index1 < len(seg1):
        s1_start, s1_end = seg1[index1]
        subtracks.append(get_subtrack(track1, s1_start, s1_end))
        index1 += 1

    while index2 < len(seg2):
        s2_start, s2_end = seg2[index2]
        subtracks.append(get_subtrack(track2, s2_start, s2_end))
        index2 += 1

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
    subtrack.parent_tid = track.track_id

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
            x, y, w, h = bbox[0:4]  # x, y is coordinate of top-left point of bounding box
            x += w / 2  # get center point
            y += h / 2  # get center point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range

def displayDist(seq2Dist, seq_name = None, isMerged=False, isSplit=False):
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
    if seq_name is None:          # Display all sequences' distance maps if no sequence is specified
        seqs = list(seq2Dist.keys())
        for i in range(len(seqs)):
            seq = seqs[i]
            Dist = seq2Dist[seq]
            # fig, ax = plt.subplots()
            # ticks = np.arange(len(Dist))
            # im, cbar = heatmap(Dist, ticks, ticks, ax=ax, cmap='binary', cbarlabel='distance')
            # texts = annotate_heatmap(im, valfmt="{x:.2f}")
            # fig.tight_layout()
            # plt.show()

            plt.imshow(Dist, cmap='binary')
            plt.colorbar()
            plt.title(seq + info)
            plt.show()
    else:
        assert seq_name in set(seq2Dist.keys())
        Dist = seq2Dist[seq_name]
        plt.imshow(Dist, cmap='binary')
        plt.colorbar()
        plt.title(seq_name + info)
        plt.show()

def getDistanceMap(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    # print("number of tracks:", len(tid2track))
    Dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        assert len(track1.times) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                # Dist[i][j] = getDistance(track1_id, track2_id, track1, track2)
                Dist[i][j] = getDistance_torch(track1_id, track2_id, track1, track2)
    return Dist

def getDistance_torch(track1_id, track2_id, track1, track2):
    """
    Calculates the cosine distance between two tracks using PyTorch for efficient computation.

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    # doesOverlap = (track1_id != track2_id)
    # if doesOverlap:
    #     track1_times = set(track1.times)
    #     track2_times = set(track2.times)
    #     doesOverlap = len(track1_times.intersection(track2_times)) > 0
    doesOverlap = False
    if (track1_id != track2_id):
        doesOverlap = set(track1.times) & set(track2.times)
    if doesOverlap:
        return 1                # make the cosine distance between two tracks maximum, max = 1
    else:
        # calculate cosine distance between two tracks based on features
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

def getDistance(track1_id, track2_id, track1, track2):
    track1_times = set(track1.times)
    track2_times = set(track2.times)

    doesOverlap = (track1_id != track2_id)
    if doesOverlap:
        doesOverlap = len(track1_times.intersection(track2_times)) > 0
        
    if doesOverlap:
        return 1                # make the cosine distance between two tracks maximum, max(cosine) = 1
    else:
        # calculate cosine distance between two tracks based on features
        track1_features = np.array(track1.features)
        track2_features = np.array(track2.features)
        count1 = len(track1.features)
        count2 = len(track2.features)

        cos_sim_Numerator = np.dot(track1_features, track2_features.T)
        track1_features_dist = np.linalg.norm(track1_features, ord=2, axis=1, keepdims=True)
        track2_features_dist = np.linalg.norm(track2_features, ord=2, axis=1, keepdims=True)
        cos_sim_Denominator = np.dot(track1_features_dist, track2_features_dist.T)
        
        cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator
        return np.sum(cos_Dist) / (count1 * count2)

def get_avg_inner_distance(embs):
    """
    Calculates the average inner cosine distance for a set of embeddings. This measure assesses the
    compactness or consistency of the embeddings within a tracklet, helping to decide if the tracklet
    should be split.

    This function is designed to handle large sets of embeddings efficiently. If the number of embeddings
    exceeds a certain threshold (e.g., 15,000), the embeddings are subsampled to fit within memory constraints
    before calculating the distance.

    Args:
        embs (list of numpy arrays): A list where each element is a numpy array representing an embedding.
                                     Each embedding has the same dimensionality.

    Returns:
        float: The average cosine distance between all pairs of embeddings in the list.
    """
    while len(embs) > 15000: # GPU memory limit
        embs = embs[1::2]
    embs = np.stack(embs)
    torch_embs = torch.from_numpy(embs).cuda()
    torch_embs = torch.nn.functional.normalize(torch_embs, dim=1)
    similarity_matrix = torch.matmul(torch_embs, torch_embs.t())
    n = similarity_matrix.shape[0]
    average_cosine_distance = 1 - (similarity_matrix.sum() - similarity_matrix.diag().sum()) / (n * (n - 1))

    return average_cosine_distance

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
    '''Debug
    assert((len(seg_1) + len(seg_2)) > 1)         # debug line, delete later
    print(seg_1)                                  # debug line, delete later
    print(seg_2)                                  # debug line, delete later
    '''
    
    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    # assert(len(subtracks) > 1)                    # debug line, delete later
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
        # check the distance between exit location of track_1 and enter location of track_2
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            # print(f"dx={dx}, dy={dy} out of range max_x_range = {max_x_range}, max_y_range  = {max_y_range}")    # debug line, delete later
            break
        else:
            subtrack_1st = subtrack_2nd
    return inSpatialRange

# FIXME: after splitting there should be more tracklets than original, current total number of tracklets after split could reduce.
def split_tracklets(tmp_trklets):
    """
    Splits each tracklet into multiple tracklets based on an internal distance threshold.

    Args:
        tmp_trklets (dict): Dictionary of tracklets to be processed.

    Returns:
        dict: New dictionary of tracklets after splitting.
    """
    
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()

    # Splitting algorithm to process every tracklets in a sequence
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < LEN_THRES:                    # NOTE: set tracklet length treshold to filter out short ones
            tracklets[tid] = trklet
        else:
            embs = np.stack(trklet.features)
            frames = np.array(trklet.times)
            bboxes = np.stack(trklet.bboxes)
            scores = np.array(trklet.scores)

            average_cosine_distance = get_avg_inner_distance(trklet.features)
            # if average_cosine_distance < inner_dist_thres
            if average_cosine_distance < INNER_DIST_THRES:      # NOTE: set inner distance threshold
                tracklets[tid] = trklet
            else:
                # print('large dist', tid, len(trklet.times), average_cosine_distance)
                recluster = False
                for k in range(2, MAX_K+1):
                    clusters = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit(embs)
                    labels = clusters.labels_
                    for i in range(k):
                        # check purity of each clustered tracklet
                        tmp_embs = embs[labels == i]
                        if get_avg_inner_distance(tmp_embs) > INNER_DIST_THRES:
                            recluster = True
                            break
                    if not recluster or k == MAX_K:
                        # clustered tracklets are pure or have reached maximum cluster limit, create new tracklets
                        for i in range(k):
                            tmp_embs = embs[labels == i]
                            tmp_frames = frames[labels == i]
                            tmp_bboxes = bboxes[labels == i]
                            tmp_scores = scores[labels == i]
                            # print('split small dist', new_id, len(tmp_embs), get_avg_inner_distance(tmp_embs))
                            assert new_id not in tmp_trklets
                            # TODO: create new tracklet object
                            tracklets[new_id] = Tracklet(new_id, tmp_frames.tolist(), tmp_scores.tolist(), tmp_bboxes.tolist(), feats=tmp_embs.tolist())
                            new_id += 1
    assert len(tracklets) >= len(tmp_trklets)
    return tracklets

def merge_tracklets(tracklets, seq_name, seq2Dist, Dist, max_x_range, max_y_range):
    seq2Dist[seq_name] = Dist                               # save all seqs distance matrix, debug line, delete later
    # displayDist(seq2Dist, seq_name, isMerged=False, isSplit=True)         # used to display Dist, debug line, delete later=

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    
    # Iterative Hierarchical Clustering
    # While there are still values (exclude diagonal) in distance matrix lower than merging distance threshold
    #   Step 1: find minimal distance for tracklet pair
    #   Step 2: merge tracklet pair
    #   Step 3: update distance matrix
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    while (np.any(Dist[non_diagonal_mask] < MERGE_DIST_THRES)):
        # Get the indices of the minimum value considering the mask
        min_index = np.argmin(Dist[non_diagonal_mask])
        min_value = np.min(Dist[non_diagonal_mask])
        
        # Translate this index to the original array's indices
        masked_indices = np.where(non_diagonal_mask)
        track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]

        # print(f"Minimum value in masked Dist: {min_value}")
        # print(f"Corresponding value in Dist using recalculated indices: {Dist[track1_idx, track2_idx]}")

        assert min_value == Dist[track1_idx, track2_idx], "Values should match!"

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        # inSpatialRange = True

        if inSpatialRange:
            track1.features += track2.features      # Note: currently we merge track 2 to track 1 without creating a new track
            track1.times += track2.times
            track1.bboxes += track2.bboxes

            # update tracklets dictionary
            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            # update distance matrix
            Dist = getDistanceMap(tracklets)
            seq2Dist[seq_name] = Dist                   # used to display Dist debug line, delete later
            # update idx2tid
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
            # update mask
            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask

    return tracklets

def merge_tracklets_new(tracklets, seq_name, seq2Dist, Dist, max_x_range, max_y_range):
    """
    Merges tracklets within a sequence based on spatial and distance constraints.

    Args:
        tracklets (dict): Dictionary of tracklets.
        seq_name (str): Name of the sequence.
        seq2Dist (dict): Dictionary to store the updated distance matrix.
        Dist (ndarray): Current distance matrix for the sequence.
        max_x_range (float): Maximum allowable x-range for merging.
        max_y_range (float): Maximum allowable y-range for merging.

    Returns:
        dict: Updated dictionary of tracklets after merging.
    """

    def merge_tracks(track1, track2):
        # Helper function to merge tracks
        track1.features.extend(track2.features)
        track1.times.extend(track2.times)
        track1.bboxes.extend(track2.bboxes)
        track1.scores.extend(track2.scores)
        return track1
    
    seq2Dist[seq_name] = Dist  # Save all seqs distance matrix for debugging
    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

    # Iterative Hierarchical Clustering
    while True:
        diagonal_mask = np.eye(len(Dist), dtype=bool)
        non_diagonal_mask = ~diagonal_mask
        if not np.any(Dist[non_diagonal_mask] < MERGE_DIST_THRES):
            break  # Exit the loop if no elements in Dist are below the threshold
        
        track1_idx, track2_idx = np.unravel_index(np.argmin(Dist[non_diagonal_mask]), Dist.shape)
        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]
        
        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        if inSpatialRange:
            # Merge track2 into track1
            tracklets[idx2tid[track1_idx]] = merge_tracks(track1, track2)
            # Update the tracklets dictionary: remove merged tracklet
            del tracklets[idx2tid[track2_idx]]
            # Recalculate the distance matrix since we have changed the number of tracklets
            Dist = getDistanceMap(tracklets)
            # Update idx2tid after modifying tracklets
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
            seq2Dist[seq_name] = Dist  # Optional: save the updated distance matrix for debugging

    return tracklets
    
def save_results(sct_output_path, tracklets):
    """
    Saves the final tracklet results into a specified path.

    Args:
        sct_output_path (str): Path where the results will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.

    """
    results = []
    for track_id, track in tracklets.items(): # add each track to results
        tid = track.track_id     # Note: it's the same as track_id
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
    
    # NOTE: uncomment to save results
    with open(sct_output_path, 'w') as f:
        f.writelines(txt_results)
    logger.info(f"save SCT results to {sct_output_path}")

def main():
    # data_path = os.path.join('..', '..')
    # seq_path = os.path.join(data_path,'Tracklets')
    seq_tracks_dir = r'C:\Users\Ciel Sun\OneDrive - UW\EE 599\SoccerNet\ByteTrack_Results\ByteTrack_Tracklets_test'
    data_path = os.path.dirname(seq_tracks_dir)
    seqs_tracks = os.listdir(seq_tracks_dir)
    process = PROCESS
    
    tracker = os.path.basename(seq_tracks_dir).split('_')[0]
    if 'ByteTrack' in seq_tracks_dir:
        tracker = 'ByteTrack'
    elif 'DeepEIoU' in seq_tracks_dir:
        tracker = 'DeepEIoU'
    else:
        assert tracker in ('ByteTrack', 'DeepEIoU',)

    if 'SportsMOT' in seq_tracks_dir:
        dataset = 'SportsMOT'
    elif 'SoccerNet' in seq_tracks_dir:
        dataset = 'SoccerNet'
    else:
        assert dataset

    seqs_tracks.sort()
    seq2Dist = dict()                       # sequence name -> distance matrix used to display Dist, debug line, delete later


    process_limit = 1000                      # debug line, delete later

    for seq_idx, seq in enumerate(seqs_tracks):
        if seq_idx >= process_limit:         # debug line, delete later
            break                           # debug line, delete later

        seq_name = seq.split('.')[0]
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}")
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)
        
        max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, SPATIAL_FACTOR)
        
        # Dist = getDistanceMap(tmp_trklets)
        # seq2Dist[seq_name] = Dist                               # save all seqs distance matrix, debug line, delete later
        # displayDist(seq2Dist, seq_name, isMerged=False, isSplit=False)         # used to display Dist, debug line, delete later

        if 'Split' in process:
            print(f"----------------Number of tracklets before splitting: {len(tmp_trklets)}----------------")
            splitTracklets = split_tracklets(tmp_trklets)
            # print(f"----------------Number of tracklets after splitting: {len(splitTracklets)}----------------")
        else:
            splitTracklets = tmp_trklets
        
        Dist = getDistanceMap(splitTracklets)

        print(f"----------------Number of tracklets before merging: {len(splitTracklets)}----------------")
        
        mergedTracklets = merge_tracklets(splitTracklets, seq_name, seq2Dist, Dist, max_x_range, max_y_range)

        print(f"----------------Number of tracklets after merging: {len(mergedTracklets)}----------------")

        sct_name = f'{tracker}_{dataset}_{process}_innerDist{INNER_DIST_THRES}_K{MAX_K}_MergeDist{MERGE_DIST_THRES}_spatialFactor{SPATIAL_FACTOR}'
        # sct_name = f'SoccertNetTest_Baseline_SCT'
        os.makedirs(os.path.join(data_path, sct_name), exist_ok=True)
        new_sct_output_path = os.path.join(data_path, sct_name, '{}.txt'.format(seq_name))
        save_results(new_sct_output_path, mergedTracklets)

    print("Done! Processed", len(seq2Dist), "sequences", f"Merged each sequence's tracks with {MERGE_DIST_THRES} as threshold")

if __name__ == "__main__":
    main()