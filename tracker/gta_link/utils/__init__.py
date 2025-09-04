# __init__.py for tracker.gta_link.utils package
import sys
import os

# Add the root directory to sys.path so we can import Tracklet
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from Tracklet import Tracklet
