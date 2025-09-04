# gta_link module within tracker package
import sys
import os

# Add the parent directory to sys.path so we can import Tracklet
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Tracklet import Tracklet

class DummyClass:
    """A generic dummy class that can accept any initialization parameters"""
    def __init__(self, *args, **kwargs):
        # Store all arguments as attributes
        for i, arg in enumerate(args):
            setattr(self, f'arg_{i}', arg)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        # Return a dummy value for any missing attribute
        return DummyClass()
    
    def __call__(self, *args, **kwargs):
        # Make the object callable
        return DummyClass(*args, **kwargs)

# Add any specific class names that might be referenced in the pickle
Tracker = DummyClass
Track = DummyClass
TrackletManager = DummyClass
Detection = DummyClass

# Use the real Tracklet class from your codebase
# Tracklet is already imported above
