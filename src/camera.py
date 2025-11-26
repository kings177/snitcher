import cv2
import time
import threading
import logging

class Camera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def start(self):
        if self.running:
            return

        self.logger.info(f"Connecting to camera: {self.rtsp_url.split('@')[-1]}") # Log safe URL
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            self.logger.error("Failed to open camera stream. Check RTSP URL and credentials.")
            # Help debug 401 errors
            if "401" in str(self.rtsp_url): # This check is naive, but we can't see the internal error easily.
                pass 
            return

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    self.logger.warning("Failed to read frame, reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.rtsp_url)
            else:
                time.sleep(1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

