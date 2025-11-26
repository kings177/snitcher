# Snitcher

A face recognition security system for Raspberry Pi 5 + Hailo-8.

## Features
- Connects to RTSP Camera
- Detects faces and identifies "Unknown" people
- Checks for loitering (time threshold)
- Sends Discord notifications with images

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `face_recognition` requires `dlib`, which compiles from source. You may need `cmake` and `build-essential`.*

2. **Configuration**
   Copy `env.example` to `.env` and edit it:
   ```bash
   cp env.example .env
   nano .env
   ```
   - Set your `RTSP_URL` (user/pass/ip)
   - Set your `DISCORD_WEBHOOK_URL`

3. **Add Known Faces**
   Add images of known people to the `known_faces/` directory.
   - `known_faces/myself.jpg`
   - `known_faces/cleber.png`
   - `known_faces/cleiton.png`

4. **Run**
   ```bash
   python src/main.py
   ```

## Hardware Acceleration (Hailo-8)
*Currently in development.*
The default implementation uses CPU-based `face_recognition` (dlib).
To use Hailo-8:
1. Ensure Hailo drivers and `hailo-platform` are installed.
2. Download `scrfd_2.5g_640x640.hef` (or similar) and `arcface_mobilefacenet.hef`.
3. (Future update will link these)
