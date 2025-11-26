import time
import logging

logger = logging.getLogger(__name__)

class LoiteringMonitor:
    def __init__(self, threshold_seconds=3.0, grace_period_seconds=2.0):
        self.threshold = threshold_seconds
        self.grace_period = grace_period_seconds
        self.first_seen = None
        self.last_seen = None
        self.triggered = False

    def update(self, has_unknown_faces):
        """
        Updates the monitor state.
        Returns True if an alert should be triggered.
        """
        now = time.time()

        if has_unknown_faces:
            self.last_seen = now
            if self.first_seen is None:
                self.first_seen = now
                logger.info("Unknown face detected. Starting timer.")

            duration = now - self.first_seen
            
            if duration > self.threshold and not self.triggered:
                logger.info(f"Loitering detected! Duration: {duration:.2f}s")
                self.triggered = True
                return True
        else:
            if self.first_seen is not None:
                if (now - self.last_seen) > self.grace_period:
                    logger.info("Unknown face lost. Resetting timer.")
                    self.first_seen = None
                    self.triggered = False
        
        return False

