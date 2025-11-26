import numpy as np
import cv2
import threading
import time
import logging
import os
import importlib

# Import post-processing utils
from .scrfd_utils import SCRFDPostProc
from .arcface_utils import face_align_norm_crop

logger = logging.getLogger(__name__)

class HailoFaceSystem:
    def __init__(self, known_faces_dir, tolerance=0.6):
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.lock = threading.Lock()
        
        # Models
        self.detector_hef = "models/scrfd.hef"
        self.recognizer_hef = "models/arcface.hef"
        
        self._check_models()
        
        # Hailo setup
        self.target = None
        self.network_group = None
        
        try:
            from hailo_platform import VDevice, HEF, InferVStreams, ConfigureParams, HailoStreamInterface
            self._hailo_platform = True
        except ImportError:
            logger.error("'hailo_platform' package not found! Make sure you are using run_snitcher.sh")
            self._hailo_platform = False
            raise

        self._init_hailo()
        
        # Post-processors
        self.scrfd = SCRFDPostProc(input_shape=(640, 640))
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def _check_models(self):
        if not os.path.exists(self.detector_hef):
            raise FileNotFoundError(f"Model not found: {self.detector_hef}")
        if not os.path.exists(self.recognizer_hef):
            raise FileNotFoundError(f"Model not found: {self.recognizer_hef}")

    def _init_hailo(self):
        if not self._hailo_platform:
            return

        from hailo_platform import VDevice, HEF, InferVStreams, ConfigureParams, HailoStreamInterface

        # Initialize VDevice
        self.vdevice = VDevice()
        
        # Detection Setup
        logger.info(f"Loading HEF: {self.detector_hef}")
        self.det_hef_obj = HEF(self.detector_hef)
        
        # Recognition Setup
        logger.info(f"Loading HEF: {self.recognizer_hef}")
        self.rec_hef_obj = HEF(self.recognizer_hef)
        
        # Configure Detection
        det_config_params = ConfigureParams.create_from_hef(hef=self.det_hef_obj, interface=HailoStreamInterface.PCIe)
        
        self.det_network_groups = self.vdevice.configure(self.det_hef_obj, det_config_params)
        self.det_network_group = self.det_network_groups[0]
        self.det_params = self.det_network_group.create_params()
        self.det_input_vstream_params = self.det_params.input_vstream_params
        self.det_output_vstream_params = self.det_params.output_vstream_params
        
        logger.info("HailoRT initialized (Detection Only for first test).")

    def _infer(self, network_group, input_params, output_params, data):
        from hailo_platform import InferVStreams
        
        with InferVStreams(network_group, input_params, output_params) as pipeline:
            input_layer_info = network_group.get_input_vstream_infos()[0]
            input_name = input_layer_info.name
            res = pipeline.infer({input_name: data})
            return res

    def process_frame(self, frame):
        # 1. Preprocess for SCRFD (640x640)
        input_size = (640, 640)
        img_h, img_w = frame.shape[:2]
        scale = min(input_size[0] / img_h, input_size[1] / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        det_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        
        # Expand dims for batch
        det_input = np.expand_dims(det_img, axis=0)
        
        # 2. Detect
        try:
            raw_det = self._infer(self.det_network_group, self.det_input_vstream_params, self.det_output_vstream_params, det_input)
            
            # 3. Postprocess
            logger.debug(f"Inference success. Keys: {raw_det.keys()}")
            return []
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []

    def load_known_faces(self):
        pass
        
    def save_face(self, frame, location, name):
        pass
