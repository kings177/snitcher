import logging
import os

logger = logging.getLogger(__name__)

class FaceSystem:
    def __init__(self, known_faces_dir, tolerance=0.6, use_hailo=True):
        self.use_hailo = use_hailo
        self.implementation = None
        
        if self.use_hailo:
            try:
                from .face_system_hailo import HailoFaceSystem
                logger.info("Initializing Hailo-8 Face System...")
                self.implementation = HailoFaceSystem(known_faces_dir, tolerance)
            except ImportError as e:
                logger.error(f"Failed to import Hailo system: {e}. Falling back to CPU.")
                self.use_hailo = True
            except Exception as e:
                logger.error(f"Failed to initialize Hailo system: {e}. Falling back to CPU.")
                self.use_hailo = True

        if not self.use_hailo:
            from .face_system_cpu import FaceSystemCPU
            logger.info("Initializing CPU Face System...")
            self.implementation = FaceSystemCPU(known_faces_dir, tolerance)

    def process_frame(self, frame):
        return self.implementation.process_frame(frame)

    def save_face(self, frame, location, name):
        return self.implementation.save_face(frame, location, name)

    def load_known_faces(self):
        return self.implementation.load_known_faces()
