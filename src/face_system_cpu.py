import face_recognition
import cv2
import os
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

class FaceSystemCPU:
    def __init__(self, known_faces_dir, tolerance=0.6):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        self.lock = threading.Lock()
        self.load_known_faces()

    def load_known_faces(self):
        with self.lock:
            self.known_face_encodings = []
            self.known_face_names = []
            
            if not os.path.exists(self.known_faces_dir):
                os.makedirs(self.known_faces_dir)
                logger.info(f"Created known faces directory: {self.known_faces_dir}")
                return

            logger.info("Loading known faces...")
            for filename in os.listdir(self.known_faces_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(self.known_faces_dir, filename)
                    try:
                        image = face_recognition.load_image_file(path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            # Use filename without extension as name
                            name = os.path.splitext(filename)[0]
                            self.known_face_names.append(name)
                            logger.info(f"Loaded face: {name}")
                        else:
                            logger.warning(f"No face found in {filename}")
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")

    def save_face(self, frame, location, name):
        """
        Saves a face from the frame to the known_faces directory and reloads.
        location: (top, right, bottom, left)
        """
        top, right, bottom, left = location
        
        # Add some padding if possible
        h, w, _ = frame.shape
        pad_h = int((bottom - top) * 0.2)
        pad_w = int((right - left) * 0.2)
        
        top = max(0, top - pad_h)
        bottom = min(h, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(w, right + pad_w)
        
        face_image = frame[top:bottom, left:right]
        
        if face_image.size == 0:
            logger.error("Failed to crop face: Empty image")
            return False

        filename = f"{name}.jpg"
        path = os.path.join(self.known_faces_dir, filename)
        
        try:
            cv2.imwrite(path, face_image)
            logger.info(f"Saved new face: {path}")
            # Reload to include the new face
            self.load_known_faces()
            return True
        except Exception as e:
            logger.error(f"Error saving face: {e}")
            return False

    def process_frame(self, frame):
        """
        Returns a list of tuples: (name, location)
        location is (top, right, bottom, left)
        """
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        results = []

        with self.lock:
            known_encodings = self.known_face_encodings
            known_names = self.known_face_names

        for face_encoding, location in zip(face_encodings, face_locations):
            # Scale location back up
            top, right, bottom, left = location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            name = "Unknown"

            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=self.tolerance)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

            results.append((name, (top, right, bottom, left)))

        return results
