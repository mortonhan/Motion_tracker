"""
Microbead motion tracker module
Implements continuous trajectory tracking based on microbead position and class
"""

from .tracker import MicroBeadTracker, YOLOTrackPipeline
from .track import Track
from .visualization import TrackVisualizer
from .matching import MatchingStrategy
from .motion_model import MotionModel
from .data_association import DataAssociation
from .utils import calculate_distance, calculate_angle, calculate_iou

try:
    from .bytetrack_tracker import ByteTrackWrapper
    BYTETRACK_AVAILABLE = True
except ImportError:
    ByteTrackWrapper = None
    BYTETRACK_AVAILABLE = False

__all__ = [
    'MicroBeadTracker',
    'YOLOTrackPipeline',
    'Track',
    'TrackVisualizer',
    'MatchingStrategy',
    'MotionModel',
    'DataAssociation',
    'calculate_distance',
    'calculate_angle',
    'calculate_iou',
    'ByteTrackWrapper',
    'BYTETRACK_AVAILABLE'
]