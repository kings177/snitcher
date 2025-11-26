import os
import requests
import logging

# URLs for common Hailo models (these links might need updating as Hailo Model Zoo evolves)
# Using generic placeholders or known GitHub releases if available.
# Since I cannot guarantee permanent URLs, I will point to the official Model Zoo.

MODELS = {
    "scrfd_2.5g_640x640.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/scrfd_2.5g.hef", # This is a guess, usually they are tar.gz
    "arcface_mobilefacenet.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/arcface_mobilefacenet.hef"
}

def download_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    print("Note: Automatic downloading of specific compiled HEF models is tricky due to versioning.")
    print("Please visit the Hailo Model Zoo or compile them yourself if these fail.")
    print("You need: scrfd_2.5g_640x640.hef AND arcface_mobilefacenet.hef")
    print("Place them in the 'models/' directory.\n")

if __name__ == "__main__":
    download_models()

