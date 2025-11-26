#!/bin/bash

# Configuration
SNITCHER_DIR=$(dirname "$(realpath "$0")")
HAILO_EXAMPLES_DIR="$HOME/hailo-rpi5-examples"
VENV_DIR="$HAILO_EXAMPLES_DIR/venv_hailo_rpi_examples"

echo "Initializing Snitcher Environment..."

# 1. Activate the Hailo Virtual Environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✅ Activated Hailo venv: $VENV_DIR"
else
    echo "❌ ERROR: Hailo venv not found at $VENV_DIR"
    echo "   Please make sure you have installed the hailo-rpi5-examples correctly."
    exit 1
fi

# 2. Add Snitcher dependencies to this venv if missing
# We check for a marker package (e.g., flask) to avoid running pip every time
if ! python -c "import flask" &> /dev/null; then
    echo "📦 Installing Snitcher dependencies into Hailo venv..."
    pip install -r "$SNITCHER_DIR/requirements.txt"
fi

# 3. Set PYTHONPATH to include Hailo examples (if needed for utils) and Snitcher
export PYTHONPATH="$HAILO_EXAMPLES_DIR:$SNITCHER_DIR:$PYTHONPATH"

# 4. Run the command passed to this script
exec "$@"

