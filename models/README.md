# Models Directory

This directory should contain the Hailo HEF model files:

## Required Models

### For Raspberry Pi 5 with Hailo-8L:
```bash
# Copy from hailo-rpi5-examples after running download_resources.sh
cp ~/hailo-rpi5-examples/resources/models/hailo8l/scrfd_2.5g.hef models/scrfd.hef
cp ~/hailo-rpi5-examples/resources/models/hailo8l/arcface_mobilefacenet_h8l.hef models/arcface.hef
```

### For Raspberry Pi with Hailo-8 (if you have the standard Hailo-8):
```bash
# Use hailo8 directory instead of hailo8l
cp ~/hailo-rpi5-examples/resources/models/hailo8/scrfd_*.hef models/scrfd.hef
cp ~/hailo-rpi5-examples/resources/models/hailo8/arcface_*.hef models/arcface.hef
```

## Model Info
- **scrfd.hef**: Face detection model (SCRFD 2.5G)
- **arcface.hef**: Face recognition model (ArcFace MobileFaceNet)

These models are not included in this repository due to size and licensing.
Download them from the official Hailo Model Zoo or use the hailo-rpi5-examples download script.

