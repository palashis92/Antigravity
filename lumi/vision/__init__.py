"""Vision Subsystem for LUMI: Camera Abstraction, Face, Object, Plant Disease, and Chess Vision."""

from .camera import CameraInterface
from .chess import ChessVision
from .face import FaceRecognitionService
from .object_detection import ObjectDetector
from .plant import PlantDiseaseDetector

__all__ = [
    "CameraInterface",
    "ChessVision",
    "FaceRecognitionService",
    "ObjectDetector",
    "PlantDiseaseDetector",
]
