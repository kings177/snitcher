import numpy as np
import cv2
import threading
import logging
import os
import pickle

from .scrfd_utils import SCRFDPostProc

logger = logging.getLogger(__name__)

class HailoFaceSystem:
    def __init__(self, known_faces_dir, tolerance=0.6):
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance  # Lower = stricter matching
        self.lock = threading.Lock()
        
        self.detector_hef = "models/scrfd.hef"
        self.recognizer_hef = "models/arcface.hef"
        
        if not os.path.exists(self.detector_hef): raise FileNotFoundError(self.detector_hef)
        if not os.path.exists(self.recognizer_hef): raise FileNotFoundError(self.recognizer_hef)
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        
        try:
            from hailo_platform import VDevice, HEF, ConfigureParams, HailoStreamInterface, InputVStreamParams, OutputVStreamParams
        except ImportError:
            raise ImportError("hailo_platform not found")

        self.vdevice = VDevice()
        
        # Detection model
        logger.info(f"Loading HEF: {self.detector_hef}")
        self.det_hef_obj = HEF(self.detector_hef)
        det_config_params = ConfigureParams.create_from_hef(hef=self.det_hef_obj, interface=HailoStreamInterface.PCIe)
        self.det_network_groups = self.vdevice.configure(self.det_hef_obj, det_config_params)
        self.det_network_group = self.det_network_groups[0]
        self.det_input_vstream_params = InputVStreamParams.make_from_network_group(self.det_network_group)
        self.det_output_vstream_params = OutputVStreamParams.make_from_network_group(self.det_network_group)
        
        # Recognition model
        logger.info(f"Loading HEF: {self.recognizer_hef}")
        self.rec_hef_obj = HEF(self.recognizer_hef)
        rec_config_params = ConfigureParams.create_from_hef(hef=self.rec_hef_obj, interface=HailoStreamInterface.PCIe)
        self.rec_network_groups = self.vdevice.configure(self.rec_hef_obj, rec_config_params)
        self.rec_network_group = self.rec_network_groups[0]
        self.rec_input_vstream_params = InputVStreamParams.make_from_network_group(self.rec_network_group)
        self.rec_output_vstream_params = OutputVStreamParams.make_from_network_group(self.rec_network_group)
        
        # ArcFace dequantization params
        self.arcface_scale = 0.01673891767859459
        self.arcface_zp = 144.0
        
        self.scrfd = SCRFDPostProc(input_shape=(640, 640), conf_thresh=0.5, nms_thresh=0.4)
        
        # Known faces database: {name: embedding}
        self.known_faces = {}
        self.embeddings_file = os.path.join(self.known_faces_dir, "embeddings.pkl")
        
        logger.info("HailoRT initialized (Detection + Recognition).")
        self.load_known_faces()

    def _infer_detection(self, data):
        from hailo_platform import InferVStreams
        with self.det_network_group.activate(network_group_params=None):
            with InferVStreams(self.det_network_group, self.det_input_vstream_params, self.det_output_vstream_params) as pipeline:
                input_name = list(self.det_input_vstream_params.keys())[0]
                return pipeline.infer({input_name: data})

    def _infer_recognition(self, data):
        from hailo_platform import InferVStreams
        with self.rec_network_group.activate(network_group_params=None):
            with InferVStreams(self.rec_network_group, self.rec_input_vstream_params, self.rec_output_vstream_params) as pipeline:
                input_name = list(self.rec_input_vstream_params.keys())[0]
                return pipeline.infer({input_name: data})

    def _get_face_embedding(self, face_img):
        """Get 512-dim embedding for a face image."""
        # Resize to 112x112 for ArcFace
        face_resized = cv2.resize(face_img, (112, 112))
        face_input = np.expand_dims(face_resized, axis=0)
        
        result = self._infer_recognition(face_input)
        
        # Get output and dequantize
        output_name = list(result.keys())[0]
        raw_embedding = result[output_name][0]  # (512,)
        embedding = (raw_embedding.astype(np.float32) - self.arcface_zp) * self.arcface_scale
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _compare_embeddings(self, emb1, emb2):
        """Compute cosine similarity between two embeddings."""
        return np.dot(emb1, emb2)

    def _find_match(self, embedding):
        """Find the best matching known face with confidence score."""
        best_match = None
        best_score = -1
        
        for name, known_embs in self.known_faces.items():
            # If multiple embeddings for this person, check all
            if isinstance(known_embs, list):
                scores = [self._compare_embeddings(embedding, emb) for emb in known_embs]
                score = max(scores)
            else:
                score = self._compare_embeddings(embedding, known_embs)
            
            if score > best_score:
                best_score = score
                best_match = name
        
        # Thresholds for categorization
        high_confidence_threshold = 0.5  # Strong match
        low_confidence_threshold = 0.35   # Uncertain match
        
        if best_score >= high_confidence_threshold:
            return best_match, best_score, "known"
        elif best_score >= low_confidence_threshold:
            return best_match, best_score, "maybe"
        else:
            return None, best_score, "unknown"

    def process_frame(self, frame):
        input_size = (640, 640)
        img_h, img_w = frame.shape[:2]
        scale = min(input_size[0] / img_h, input_size[1] / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        det_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        det_input = np.expand_dims(det_img, axis=0)
        
        try:
            raw_det = self._infer_detection(det_input)
            
            scores_list = [
                raw_det["scrfd_2_5g/conv42"],
                raw_det["scrfd_2_5g/conv49"],
                raw_det["scrfd_2_5g/conv55"]
            ]
            bboxes_list = [
                raw_det["scrfd_2_5g/conv43"],
                raw_det["scrfd_2_5g/conv50"],
                raw_det["scrfd_2_5g/conv56"]
            ]
            kpss_list = [
                raw_det["scrfd_2_5g/conv44"],
                raw_det["scrfd_2_5g/conv51"],
                raw_det["scrfd_2_5g/conv57"]
            ]
            
            bboxes, kpss, scores = self.scrfd.postprocess(scores_list, bboxes_list, kpss_list, scale=scale)
            
            results = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox[:4].astype(int)
                
                # Crop face for recognition
                face_crop = frame[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]
                
                if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                    try:
                        embedding = self._get_face_embedding(face_crop)
                        match_name, confidence, category = self._find_match(embedding)
                        
                        if category == "known":
                            name = f"{match_name} ({confidence:.0%})"
                        elif category == "maybe":
                            name = f"Maybe {match_name}? ({confidence:.0%})"
                        else:
                            name = "Unknown"
                    except Exception as e:
                        logger.debug(f"Recognition error: {e}")
                        name = "Unknown"
                        category = "unknown"
                        confidence = 0.0
                else:
                    name = "Unknown"
                    category = "unknown"
                    confidence = 0.0
                
                # Format: (name, (top, right, bottom, left), category, confidence)
                results.append((name, (y1, x2, y2, x1), category, confidence))
                
            return results

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []

    def load_known_faces(self):
        """Load embeddings from file."""
        self.known_faces = {}
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces: {list(self.known_faces.keys())}")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        else:
            logger.info("No saved embeddings found.")

    def _save_embeddings(self):
        """Save embeddings to file."""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.info(f"Saved {len(self.known_faces)} face embeddings.")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def save_face(self, frame, location, name, multi_sample=True, num_samples=10, sample_delay=0.3):
        """
        Save a face and its embeddings.
        
        Args:
            frame: Current video frame
            location: Face bounding box (top, right, bottom, left)
            name: Person's name
            multi_sample: If True, will request multiple samples via callback
            num_samples: Number of samples to collect
            sample_delay: Delay between samples (seconds)
        """
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_name:
            safe_name = "face"
        
        # Store the request for multi-sampling
        # The web interface will need to call this multiple times
        # Or we can store state and collect samples over time
        
        # For now, we'll save the current frame and embedding
        # And store as a list to support multiple embeddings per person
        
        top, right, bottom, left = location
        h, w = frame.shape[:2]
        pad_h = int((bottom - top) * 0.3)
        pad_w = int((right - left) * 0.3)
        
        top = max(0, top - pad_h)
        bottom = min(h, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(w, right + pad_w)
        
        face_image = frame[top:bottom, left:right]
        
        if face_image.size == 0:
            logger.error("Failed to crop face: Empty image")
            return False

        try:
            # Generate embedding
            embedding = self._get_face_embedding(face_image)
            
            # Add to existing embeddings or create new list
            if safe_name in self.known_faces:
                if not isinstance(self.known_faces[safe_name], list):
                    # Convert single embedding to list
                    self.known_faces[safe_name] = [self.known_faces[safe_name]]
                self.known_faces[safe_name].append(embedding)
                logger.info(f"Added embedding #{len(self.known_faces[safe_name])} for: {safe_name}")
            else:
                # New person - store as list for consistency
                self.known_faces[safe_name] = [embedding]
                logger.info(f"Created new face entry for: {safe_name}")
            
            # Save image (overwrite or numbered?)
            num_existing = len([f for f in os.listdir(self.known_faces_dir) if f.startswith(safe_name)])
            filename = f"{safe_name}_{num_existing}.jpg"
            path = os.path.join(self.known_faces_dir, filename)
            cv2.imwrite(path, face_image)
            logger.info(f"Saved face image: {path}")
            
            self._save_embeddings()
            return True
            
        except Exception as e:
            logger.error(f"Error saving face: {e}")
            return False
