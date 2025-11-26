import os
import cv2
import numpy as np
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestHailo")

def test_hailo():
    # 1. Check imports
    try:
        import hailo_platform
        print(f"SUCCESS: hailo_platform imported.")
    except ImportError:
        print("FAILURE: hailo_platform NOT found.")
        return

    # 2. Check Models
    if not os.path.exists("models/scrfd.hef"):
        print("FAILURE: models/scrfd.hef not found.")
        return
    print("SUCCESS: Models found.")

    # 3. Initialize VDevice
    try:
        from hailo_platform import VDevice, HEF, ConfigureParams, HailoStreamInterface
        target = VDevice()
        ids = target.get_physical_devices_ids()
        print(f"SUCCESS: VDevice initialized. Device IDs: {ids}")
    except Exception as e:
        print(f"FAILURE: VDevice init failed: {e}")
        return

    # 4. Load HEF
    try:
        hef = HEF("models/scrfd.hef")
        print("SUCCESS: SCRFD HEF loaded.")
        
        # Configure
        # Use explicit Enum for PCIe
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        
        network_groups = target.configure(hef, params)
        print(f"SUCCESS: Network configured. Groups: {len(network_groups)}")
        
    except Exception as e:
        print(f"FAILURE: HEF Load/Config failed: {e}")
        return

    print("\nALL SYSTEM CHECKS PASSED. You can try running with USE_HAILO=true.")

if __name__ == "__main__":
    test_hailo()
