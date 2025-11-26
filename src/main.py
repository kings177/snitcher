import cv2
import time
import logging
import sys
import os

# Add src to path if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RTSP_URL, DISCORD_WEBHOOK_URL, KNOWN_FACES_DIR
from src.camera import Camera
from src.face_system import FaceSystem
from src.monitor import LoiteringMonitor
from src.notifier import DiscordNotifier
from src.web_stream import WebStreamer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")

def main():
    logger.info("Starting Snitcher...")
    
    # Initialize components
    camera = Camera(RTSP_URL)
    
    logger.info(f"Initializing Face System with directory: {KNOWN_FACES_DIR}")
    # Try to use Hailo if configured, otherwise CPU
    use_hailo = os.getenv("USE_HAILO", "false").lower() == "true"
    face_system = FaceSystem(KNOWN_FACES_DIR, use_hailo=use_hailo)
    
    monitor = LoiteringMonitor(threshold_seconds=5.0, grace_period_seconds=2.0)
    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    web_stream = WebStreamer(face_system=face_system)

    camera.start()
    web_stream.start()
    
    # Wait for camera to warm up
    time.sleep(2)

    logger.info("System ready. Press Ctrl+C to stop.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.warning("No frame received")
                time.sleep(0.1)
                continue

            # Process frame
            results = face_system.process_frame(frame)
            
            # Keep a raw copy for saving faces (clean image)
            raw_frame = frame.copy()

            # Check for unknown faces
            has_unknown = False
            for name, (top, right, bottom, left) in results:
                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if name == "Unknown":
                    has_unknown = True

            # Update web stream
            web_stream.update_frame(frame, raw_frame, results)

            # Update monitor
            if monitor.update(has_unknown):
                logger.info("ALARM! Sending notification.")
                notifier.send_alert(frame, "⚠️ Intruder Detected! ⚠️")

            # Display (if possible)
            # You can disable this on headless systems
            try:
                cv2.imshow("Snitcher Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except (cv2.error, AttributeError):
                pass # Headless or no display

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        camera.stop()
        try:
            cv2.destroyAllWindows()
        except (cv2.error, AttributeError):
            pass

if __name__ == "__main__":
    main()

