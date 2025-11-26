# Snitcher 🔍

A face recognition security system for Raspberry Pi 5 with Hailo-8 AI acceleration. Monitors your RTSP camera, detects faces, recognizes known people, and sends Discord alerts when unknown visitors loiter.

## ✨ Features
- 🎥 **RTSP Camera Support** - Works with any IP camera (tested with Intelbras VIP 1230 B)
- ⚡ **Hailo-8 Acceleration** - Fast face detection using SCRFD and ArcFace models
- 🧠 **Face Recognition** - Distinguish between known people and strangers
- ⏱️ **Loitering Detection** - Only alerts if someone stays in view (not just passing by)
- 💬 **Discord Notifications** - Get alerts with images when intruders are detected
- 🌐 **Web Interface** - Live view and face management via browser
- 🔄 **CPU Fallback** - Automatically falls back to CPU if Hailo unavailable

## 📋 Hardware Requirements
- Raspberry Pi 5
- Hailo-8 or Hailo-8L AI Accelerator
- IP Camera with RTSP support (PoE recommended)

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the Hailo software stack installed:
```bash
# If not already installed, follow the official Hailo guide:
# https://github.com/hailo-ai/hailo-rpi5-examples
```

### 2. Clone and Setup
```bash
git clone https://github.com/yourusername/snitcher.git
cd snitcher
chmod +x run_snitcher.sh
```

### 3. Download Models
```bash
# Copy models from hailo-rpi5-examples (after running their download_resources.sh)
mkdir -p models
cp ~/hailo-rpi5-examples/resources/models/hailo8l/scrfd_2.5g.hef models/scrfd.hef
cp ~/hailo-rpi5-examples/resources/models/hailo8l/arcface_mobilefacenet_h8l.hef models/arcface.hef
```

### 4. Configure
```bash
cp env.example .env
nano .env
```

Edit the following:
- `RTSP_URL`: Your camera's RTSP URL (format: `rtsp://user:pass@ip:port/path`)
  - **Note**: URL-encode special characters in passwords (e.g., `#` → `%23`)
- `DISCORD_WEBHOOK_URL`: Your Discord webhook URL (optional)
- `USE_HAILO`: Set to `true` to enable Hailo acceleration

### 5. Run
```bash
./run_snitcher.sh python src/main.py
```

### 6. Access Web Interface
Open `http://<RASPBERRY_PI_IP>:5000` in your browser.

## 👤 Adding Known Faces

1. Open the web interface
2. Stand in front of the camera until a **RED box** appears around your face
3. Enter your name in the text field
4. Click **"Save Detected Face"**
5. The box should turn **GREEN** and display your name on subsequent appearances

## 📁 Project Structure
```
snitcher/
├── src/
│   ├── main.py                # Application entry point
│   ├── camera.py              # RTSP stream handler with auto-reconnect
│   ├── face_system.py         # Face system wrapper (Hailo/CPU selector)
│   ├── face_system_hailo.py   # Hailo-8 accelerated implementation
│   ├── face_system_cpu.py     # CPU fallback (face_recognition lib)
│   ├── scrfd_utils.py         # SCRFD post-processing & dequantization
│   ├── arcface_utils.py       # Face alignment utilities
│   ├── monitor.py             # Loitering detection logic
│   ├── notifier.py            # Discord webhook notifications
│   ├── web_stream.py          # Flask web server for live view
│   └── config.py              # Environment configuration loader
├── models/                    # HEF model files (not included)
├── known_faces/               # Saved face images and embeddings
├── tools/                     # Utility scripts
│   ├── test_cam.py           # Test RTSP connection
│   └── test_hailo_env.py     # Verify Hailo environment
├── run_snitcher.sh           # Environment wrapper script
├── requirements.txt          # Python dependencies
├── env.example               # Example configuration
└── README.md                 # This file
```

## 🛠️ Troubleshooting

### RTSP Connection Issues
Run the camera test:
```bash
./run_snitcher.sh python tools/test_cam.py
```

Common issues:
- **401 Unauthorized**: Wrong username/password or special characters not URL-encoded
- **Connection timeout**: Wrong IP or camera unreachable
- **No frames**: Wrong channel/subtype in URL

### Hailo Not Working
Run the Hailo test:
```bash
./run_snitcher.sh python tools/test_hailo_env.py
```

If it fails:
- Ensure Hailo drivers are installed
- Check that models exist in `models/` directory
- Verify you're using `run_snitcher.sh` (not direct python)

### Qt/Display Errors
These are normal on headless systems and are safely ignored. The web interface still works.

## ⚙️ Configuration Options

### Environment Variables (`.env`)
- `RTSP_URL`: Camera RTSP stream URL
- `DISCORD_WEBHOOK_URL`: Discord webhook for notifications (optional)
- `KNOWN_FACES_DIR`: Directory for saved faces (default: `known_faces`)
- `USE_HAILO`: Enable Hailo acceleration (`true`/`false`)

### Adjustable Parameters
Edit in `src/main.py`:
- `threshold_seconds=5.0`: How long someone must be present before alerting
- `grace_period_seconds=2.0`: Tolerance for brief occlusions
- `conf_thresh=0.5`: Face detection confidence threshold (in `face_system_hailo.py`)
- `tolerance=0.6`: Face recognition strictness (lower = stricter)

## 📊 Performance
- **Hailo-8L**: ~20-30 FPS on 1080p stream
- **CPU Fallback**: ~2-5 FPS on Pi 5

## 🔒 Security Notes
- Never commit your `.env` file (it's in `.gitignore`)
- Face embeddings are stored locally in `known_faces/embeddings.pkl`
- No data is sent to external services except Discord (if configured)

## 📜 License
MIT

## 🙏 Credits
- Uses Hailo's SCRFD and ArcFace models
- Built on the hailo-rpi5-examples framework
- OpenCV for video processing
- Flask for web interface

## 🤝 Contributing
Pull requests welcome! This project is meant to be reusable by anyone with similar hardware.

## ⚠️ Disclaimer
This is a DIY security project. Do not rely on it as your sole security measure. Always comply with local privacy and surveillance laws when recording video.
