import os
from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
KNOWN_FACES_DIR = os.getenv("KNOWN_FACES_DIR", "known_faces")

# Validation
if not RTSP_URL:
    print("Warning: RTSP_URL not set in .env")

