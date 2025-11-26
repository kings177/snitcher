#!/bin/bash
# Diagnostic script to check file sync status

echo "=== Checking local files ==="
echo "face_system.py: $(wc -l src/face_system.py | awk '{print $1}') lines"
echo "face_system_hailo.py: $(wc -l src/face_system_hailo.py | awk '{print $1}') lines"

echo ""
echo "=== Checking remote files ==="
echo "face_system.py: $(ssh pi1 'wc -l snitcher/src/face_system.py 2>/dev/null | awk "{print \$1}"') lines"
echo "face_system_hailo.py: $(ssh pi1 'wc -l snitcher/src/face_system_hailo.py 2>/dev/null | awk "{print \$1}"') lines"

echo ""
echo "=== Checking for None check in face_system.py ==="
echo "Local:"
grep -n "if self.implementation is None" src/face_system.py || echo "  NOT FOUND"
echo "Remote:"
ssh pi1 'grep -n "if self.implementation is None" snitcher/src/face_system.py' || echo "  NOT FOUND"

echo ""
echo "=== Checking for InputVStreamParams in face_system_hailo.py ==="
echo "Local:"
grep -n "InputVStreamParams.make_from_network_group" src/face_system_hailo.py | head -2
echo "Remote:"
ssh pi1 'grep -n "InputVStreamParams.make_from_network_group" snitcher/src/face_system_hailo.py' | head -2

