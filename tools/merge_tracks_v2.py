import numpy as np
import os
import torch
import pickle
import sys
sys.path.append('..')
print(sys.path)
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from loguru import logger
from tracker.Deep_EIoU import STrack
import shutil # used for copying files
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering

class Tracklet:
    def __init__(self, track_id, frames, scores, bboxes, feats=None):
        '''
        Initialize the Tracklet with IDs, times, scores, bounding boxes, and optional features.
        - frames, scores can be lists or single elements.
        - bboxes can be a single list of 4 elements or a list of lists where each sublist has 4 elements.
        - feats should be a list of numpy arrays each of shape (512,) or None.
        '''
        self.track_id = track_id
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

# TODO:
# 1. Add comments to functions and hyperparameters
# 2. Test code
# 3. Delete unused lines/functions

# Define hyperparameters for merging tracklets
SPATIAL_FACTOR = 1        # spatial constraint factor restricting spatial distance between two targets to be merged
SELF_DIST_FACTOR = 2      # Multiplier for the self-distance of a track to set a dynamic merging threshold.
MAX_DIST = 0.4            # Define the upper bound for the dynamic merging threshold, ensuring it stays within a reasonable range.
MIN_DIST = 0.4            # Define the lower bound

# DEFINE hyperparameters for splitting tracklets
LEN_THRES = 100
MAX_K = 5
INNER_DIST_THRES = 0.2

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
    Find consecutive segments in a list and store their starting and ending indices.

    Args:
    lst (list): A list of elements.

    Returns:
    segments: a list of tuples representing the starting and ending indices of each segment 
                             with respect to the original list.
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
    Process segments to generate valid subtracks based on temporal constraints.

    Args:
    seg1 (list): List of segments for track 1.
    seg2 (list): List of segments for track 2.
    track1 (STrack): Track 1 object containing time intervals and bounding boxes.
    track2 (STrack): Track 2 object containing time intervals and bounding boxes.

    Returns:
    list: A list of temporally sorted valid subtracks extracted from track 1 and track 2 based on the provided segments.
    """
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        '''Optionally eliminate false positive subtracks
        if (s1_end - s1_start + 1) < 30:
            seg1.pop(0)  # Remove the first element from seg1
            continue
        if (s2_end - s2_start + 1) < 30:
            seg2.pop(0)  # Remove the first element from seg2
            continue
        '''
        
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2

        subtrack_1 = get_subtrack(track1, s1_start, s1_end)  # Extract subtrack from track 1
        subtrack_2 = get_subtrack(track2, s2_start, s2_end)  # Extract subtrack from track 2
        s1_startFrame = track1.times[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.times[s2_start]  # Get the starting frame of subtrack 2
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
        subtracks.append(get_subtrack(track_remain, s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks  # Return the list of valid subtracks sorted ascending temporally

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
    subtrack = STrack(None, None)
    subtrack.times = track.times[s_start : s_end + 1]
    subtrack.bboxes = track.bboxes[s_start : s_end + 1]
    subtrack.parent_tid = track.track_id

    return subtrack

def get_spatial_constraints(tid2track, factor):
    """
    Obtain parameters for spatial distance constraints based on all locations of bounding boxes.

    Args:
    tid2track (dict): A dictionary mapping track IDs to their corresponding track objects.
    factor (float): A hyperparameter to downscale the max range of x and y

    Returns:
    tuple: A tuple containing the maximal x and y range obtained from the bounding boxes.
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
    '''
    (dict) seq2Dist: seq_name -> distance map
    (str) seq_name: name of sequence
    (bool) isMerged: True if the displayed distance matrix results from merging, False otherwise.
    '''
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
    track1_times = set(track1.times)
    track2_times = set(track2.times)

    doesOverlap = (track1_id != track2_id)
    if doesOverlap:
        doesOverlap = len(track1_times.intersection(track2_times)) > 0
        
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
    
def sort_and_filter_distance_array(arr, threshold, self_idx):
    # return filtered idx to distance dictionary sorted by ascendingly distance values
    filtered_dict = {}
    for i, distance in enumerate(arr):
        if distance < threshold and i != self_idx:
            filtered_dict[i] = distance
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1]))
    return sorted_dict

def get_avg_inner_distance(embs):
    '''
    This function computes a tracklet's embbedings' inner consine distance.
    (list) embs: list of embbedings in numpy array (512,)
    '''
    while len(embs) > 15000: # GPU memory limit
        embs = embs[1::2]
    embs = np.stack(embs)
    torch_embs = torch.from_numpy(embs).cuda()
    torch_embs = torch.nn.functional.normalize(torch_embs, dim=1)
    similarity_matrix = torch.matmul(torch_embs, torch_embs.t())
    n = similarity_matrix.shape[0]
    average_cosine_distance = 1 - (similarity_matrix.sum() - similarity_matrix.diag().sum()) / (n * (n - 1))

    return average_cosine_distance

def main():
    # data_path = os.path.join('..', '..')
    # seq_path = os.path.join(data_path,'Tracklets')
    seq_tracks_path = r'C:\Users\Ciel Sun\OneDrive - UW\EE 599\SoccerNet\tracking-2023\test_Seq_Tracklets'
    data_path = os.path.dirname(seq_tracks_path)
    seqs_tracks = os.listdir(seq_tracks_path)
    seqs_tracks.sort()
    seq2Dist = dict()                       # sequence name -> distance matrix used to display Dist, debug line, delete later


    # process_limit = 2                      # debug line, delete later

    for seq_idx, seq in enumerate(seqs_tracks):
        # if seq_idx >= process_limit:         # debug line, delete later
        #     break                           # debug line, delete later

        seq_name = seq.split('.')[0]
        logger.info(f"Processing seq {seq_idx+1} / {len(seqs_tracks)}")
        with open(os.path.join(seq_tracks_path, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)     # dict(key:track id, value:tracklet)
        
        max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, SPATIAL_FACTOR)
        
        # Dist = getDistanceMap(tmp_trklets)
        # seq2Dist[seq_name] = Dist                               # save all seqs distance matrix, debug line, delete later
        # displayDist(seq2Dist, seq_name, isMerged=False, isSplit=False)         # used to display Dist, debug line, delete later
        
        new_id = max(tmp_trklets.keys()) + 1
        tracklets = defaultdict()

        # Splitting algorithm to process every tracklets in a sequence
        for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
            trklet = tmp_trklets[tid]
            if len(trklet.times) > LEN_THRES:                    # NOTE: set tracklet length treshold to filter out short ones
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
        
        ''' Merge Implementation Pseudocode
        while not done:
            for each row in distance matrix Dist:
                idx2dist: sort from smallest to largest dist
                for each element in row that is smaller than dist threshold, starting from smallest element
                    if element is also minimum in column
                        merge corresponding two tracks
                        update distance matrix Dist
                        update tid2track
                        update idx2track_id
                        break current loop
            if merged tracks
                break current loop and start from first row again
            else if reached last row
                done
        Alternatively, iteratively merge tracks one pair at a time with the minimum distance
        '''
        Dist = getDistanceMap(tracklets)

        seq2Dist[seq_name] = Dist                               # save all seqs distance matrix, debug line, delete later
        # displayDist(seq2Dist, seq_name, isMerged=False, isSplit=True)         # used to display Dist, debug line, delete later


        idx2tid = {idx: track_id for idx, track_id in enumerate(tracklets.keys())}
        
        # max_x_range, max_y_range = get_spatial_constraints(tracklets, SPATIAL_FACTOR)

        done = False
        
        while (not done):       # continuously merge tracks and update idx2tid, Dist
            '''Debug
            # print(idx2tid)                          # debug line, delete later
            # print("Dist:")                          # debug line, delete later
            # print(np.around(Dist, decimals=2))      # debug line, delete later
            '''
            for i, tid in idx2tid.items():
                # get candidate tracks for merging with current track
                self_dist = Dist[i][i]

                threshold = min(max(self_dist * SELF_DIST_FACTOR, MIN_DIST), MAX_DIST)                     # Three Hyperparameters
                logger.info(f"Merging distance threshold: {threshold:.3f} | Self distance: {self_dist:.3f} | Self disntance factor: {SELF_DIST_FACTOR:.3f}")
                # logger.info(f'Merge threshold for Track_id {tid}: {threshold:.3f} scaled from self distance {self_dist:.3f}')
                
                # Get merge candidates for current track ranked on distance low to high
                merge_idx2dist = sort_and_filter_distance_array(Dist[i], threshold, i)
                ''' Debug
                # print("Merge candidates for track i={}: idx -> dist".format(i))  # debug line, delete later
                # print(merge_idx2dist)                                            # debug line, delete later
                # done = True                                                      # debug line, delete later
                # break                                                            # debug line, delete later
                '''
                didMerge = False

                # Iterate through merge candidates
                for j, dist in merge_idx2dist.items():    # merge i and j if they are best match
                    # check if current candidate track has better tracks to merge with
                    column_min = np.min([d for col, d in enumerate(Dist[j]) if col != j])
                    # print("Column {} min: {}".format(j, column_min))             # debug line, delete later
                    if dist <= column_min:
                        # merge track at index i and track at index j
                        tid_1 = idx2tid[i]
                        tid_2 = idx2tid[j]
                        # print("Merge tracks i={} and j={}".format(tid_1, tid_2)) # debug line, delete later
                        track_1 = tracklets[tid_1]
                        track_2 = tracklets[tid_2]

                        '''Enforce spatial constraint
                              do not merge two tracks if two temporally adjacent subtracks of track1 and track2 
                              have spatial distance in x and y exceedindg threshold range
                              '''
                        inSpatialRange = True
                        seg_1 = find_consecutive_segments(track_1.times)
                        seg_2 = find_consecutive_segments(track_2.times)
                        '''Debug
                        assert((len(seg_1) + len(seg_2)) > 1)         # debug line, delete later
                        print(seg_1)                                  # debug line, delete later
                        print(seg_2)                                  # debug line, delete later
                        '''
                        
                        subtracks = query_subtracks(seg_1, seg_2, track_1, track_2)
                        # assert(len(subtracks) > 1)                    # debug line, delete later
                        subtrack_1st = subtracks.pop(0)
                        while subtracks:
                            subtrack_2nd = subtracks.pop(0)
                            if subtrack_1st.parent_tid == subtrack_2nd.parent_tid:
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
                        
                        if inSpatialRange:
                            track_1.features += track_2.features      # Note: currently we merge track 2 to track 1 without creating a new track
                            track_1.times += track_2.times
                            track_1.bboxes += track_2.bboxes

                            didMerge = True

                            # update tid2track dictionary
                            tracklets[tid_1] = track_1
                            tracklets.pop(tid_2)

                            # update distance matrix
                            Dist = getDistanceMap(tracklets)

                            seq2Dist[seq_name] = Dist                   # used to display Dist debug line, delete later

                            # update idx2tid
                            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

                            break
                if didMerge:                   # if merged tracks, start from the first track again
                    break
                elif i == len(idx2tid) - 1:    # idx2tid is not changed in the loop due to no merging, we are done after processing last track
                    done = True
                    # displayDist(seq2Dist, seq_name, isMerged=True, isSplit=True)         # used to display Dist, debug line, delete later

        sct_name = f'SoccerNetTest_SplitConnect_SCT_distThreshRange_{MIN_DIST}-{MAX_DIST}_selfDistFactor-{SELF_DIST_FACTOR}_spatialFactor_{SPATIAL_FACTOR}'
        os.makedirs(os.path.join(data_path, sct_name), exist_ok=True)
        new_sct_output_path = os.path.join(data_path, sct_name, '{}.txt'.format(seq_name))

        # save new Single Camera Tracking (SCT) results for evaluation after merging tracks
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
        with open(new_sct_output_path, 'w') as f:
            f.writelines(txt_results)
        logger.info(f"save SCT results to {new_sct_output_path}")

    print("Done! Processed", len(seq2Dist), "sequences", f"Merged each sequence's tracks with {MIN_DIST} <= self_dist * 2 <= {MAX_DIST} as threshold")

if __name__ == "__main__":
    main()