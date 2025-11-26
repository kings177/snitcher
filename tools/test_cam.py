import cv2
import os
import sys
from urllib.parse import quote_plus

# Add parent dir to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.config import RTSP_URL
except ImportError:
    print("Could not import config. Make sure you have a .env file.")
    RTSP_URL = None

def test_connection():
    if not RTSP_URL:
        print("Error: RTSP_URL not found in environment.")
        return

    print(f"Testing connection to: {RTSP_URL.split('@')[-1]}") # Mask credentials
    
    # Check for special characters in the raw string that might need encoding
    # This is a heuristic, assuming the structure rtsp://user:pass@host
    try:
        if '@' in RTSP_URL:
            credentials = RTSP_URL.split('rtsp://')[1].split('@')[0]
            if ':' in credentials:
                user, password = credentials.split(':', 1)
                print(f"DEBUG: Username detected as: {user}")
                # We won't print password, but we can check if it has unencoded chars
                special_chars = ['#', '/', '?', '&', '=', '%']
                if any(c in password for c in special_chars) and '%' not in password:
                     print("\nWARNING: Your password might contain special characters that are not URL encoded.")
                     print("Example: '#' should be '%23'.")
    except Exception:
        pass

    cap = cv2.VideoCapture(RTSP_URL)
    
    if cap.isOpened():
        print("SUCCESS: Camera connected!")
        ret, frame = cap.read()
        if ret:
            print(f"Frame received: {frame.shape}")
            cv2.imwrite("test_capture.jpg", frame)
            print("Saved test_capture.jpg")
        else:
            print("Connected, but failed to read frame.")
    else:
        print("FAILURE: Could not open stream.")
        print("Common causes:")
        print("1. Wrong Username/Password")
        print("2. Password contains special characters (needs URL encoding)")
        print("3. Camera IP is unreachable")
        print("4. Wrong Channel/Subtype in URL")

    cap.release()

if __name__ == "__main__":
    test_connection()

