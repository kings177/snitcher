import cv2
import threading
from flask import Flask, Response, render_template_string, request, jsonify
import time
import logging

logger = logging.getLogger(__name__)

class WebStreamer:
    def __init__(self, face_system=None, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.frame = None
        self.raw_frame = None # Frame without boxes for saving
        self.current_results = [] # Store detection results
        self.lock = threading.Lock()
        self.running = False
        self.face_system = face_system

        # Register routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/api/status', 'status', self.get_status, methods=['GET'])
        self.app.add_url_rule('/api/save_face', 'save_face', self.save_face, methods=['POST'])

    def update_frame(self, frame, raw_frame, results):
        with self.lock:
            self.frame = frame.copy()
            self.raw_frame = raw_frame.copy()
            self.current_results = results

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            
            # Encode frame to jpg
            ret, buffer = cv2.imencode('.jpg', self.frame)
            if not ret:
                return None
            return buffer.tobytes()

    def generate(self):
        while True:
            frame_bytes = self.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033) # ~30 FPS

    def get_status(self):
        with self.lock:
            # Check if there is an unknown face
            unknown_found = False
            for name, _ in self.current_results:
                if name == "Unknown":
                    unknown_found = True
                    break
            return jsonify({'unknown_found': unknown_found})

    def save_face(self):
        if not self.face_system:
            return jsonify({'error': 'Face system not available'}), 500

        data = request.json
        name = data.get('name')
        if not name:
            return jsonify({'error': 'Name is required'}), 400

        with self.lock:
            if self.raw_frame is None:
                 return jsonify({'error': 'No frame available'}), 400
            
            # Find the first unknown face to save
            target_location = None
            for fname, location in self.current_results:
                if fname == "Unknown":
                    target_location = location
                    break
            
            if not target_location:
                 return jsonify({'error': 'No unknown face detected to save'}), 404
            
            # Use raw frame to save clean image
            success = self.face_system.save_face(self.raw_frame, target_location, name)

        if success:
            return jsonify({'success': True, 'message': f'Saved face for {name}'})
        else:
            return jsonify({'error': 'Failed to save face'}), 500

    def index(self):
        return render_template_string("""
            <html>
            <head>
                <title>Camera Live View</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { background: #111; color: white; font-family: sans-serif; text-align: center; padding: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    img { border: 2px solid #444; max-width: 100%; height: auto; margin-bottom: 20px; }
                    .controls { background: #222; padding: 20px; border-radius: 8px; margin-top: 20px; }
                    input[type="text"] { padding: 10px; border-radius: 4px; border: 1px solid #444; background: #333; color: white; }
                    button { padding: 10px 20px; border-radius: 4px; border: none; background: #007bff; color: white; cursor: pointer; }
                    button:disabled { background: #555; cursor: not-allowed; }
                    .status { margin-top: 10px; font-weight: bold; }
                    .status.detected { color: #ff4444; }
                </style>
                <script>
                    function checkStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                const btn = document.getElementById('saveBtn');
                                const status = document.getElementById('statusText');
                                if (data.unknown_found) {
                                    btn.disabled = false;
                                    status.innerText = "Unknown Face Detected!";
                                    status.className = "status detected";
                                } else {
                                    btn.disabled = true;
                                    status.innerText = "No unknown faces";
                                    status.className = "status";
                                }
                            });
                    }

                    function saveFace() {
                        const name = document.getElementById('nameInput').value;
                        if (!name) {
                            alert("Please enter a name");
                            return;
                        }
                        
                        fetch('/api/save_face', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ name: name })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                alert(data.message);
                                document.getElementById('nameInput').value = '';
                            } else {
                                alert("Error: " + data.error);
                            }
                        });
                    }

                    setInterval(checkStatus, 1000);
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Snitcher Live Feed</h1>
                    <img src="{{ url_for('video_feed') }}">
                    
                    <div class="controls">
                        <h3>Add New Face</h3>
                        <p>Wait for a RED box (Unknown) to appear, then enter a name and click Save.</p>
                        <div id="statusText" class="status">Waiting...</div>
                        <br>
                        <input type="text" id="nameInput" placeholder="Enter Name">
                        <button id="saveBtn" onclick="saveFace()" disabled>Save Detected Face</button>
                    </div>
                </div>
            </body>
            </html>
        """)

    def video_feed(self):
        return Response(self.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def start(self):
        self.running = True
        # Run Flask in a separate thread
        thread = threading.Thread(target=self._run_flask, daemon=True)
        thread.start()
        logger.info(f"Web streamer started at http://{self.host}:{self.port}")

    def _run_flask(self):
        # Disable Flask logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
