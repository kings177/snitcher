import requests
import cv2
import logging
import time
from io import BytesIO

logger = logging.getLogger(__name__)

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.last_notification_time = 0
        self.cooldown = 60  # Seconds between notifications

    def send_alert(self, image, message="Unknown person detected!"):
        if not self.webhook_url:
            logger.warning("No Discord Webhook URL configured.")
            return

        if time.time() - self.last_notification_time < self.cooldown:
            logger.info("Notification cooldown active. Skipping.")
            return

        logger.info("Sending Discord notification...")
        
        # Encode image to jpg
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            logger.error("Failed to encode image for Discord")
            return

        image_bytes = BytesIO(encoded_image.tobytes())
        
        files = {
            'file': ('alert.jpg', image_bytes, 'image/jpeg')
        }
        data = {
            'content': message
        }

        try:
            response = requests.post(self.webhook_url, data=data, files=files)
            if response.status_code == 200 or response.status_code == 204:
                logger.info("Discord notification sent successfully.")
                self.last_notification_time = time.time()
            else:
                logger.error(f"Failed to send Discord notification: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")

