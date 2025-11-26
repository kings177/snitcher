import cv2
import numpy as np
from skimage import transform as trans

def face_align_norm_crop(img, landmark):
    """
    Align, normalize and crop face using 5 landmarks.
    landmark: 5x2
    """
    assert landmark.shape == (5, 2)
    
    # ArcFace standard landmarks (112x112)
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
        
    src[:, 0] += 8.0 # Shift x if needed (ArcFace models vary, usually 112x112 uses this)

    tform = trans.SimilarityTransform()
    tform.estimate(landmark, src)
    M = tform.params[0:2, :]
    
    # Warped image
    warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return warped

