import numpy as np

def distance2bbox(points, distance, max_shape=None):
    """Decode bounding box based on distances."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode keypoints based on distances."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape((-1, 5, 2))

class SCRFDPostProc:
    def __init__(self, input_shape=(640, 640), conf_thresh=0.5, nms_thresh=0.4):
        self.input_shape = input_shape
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.center_cache = {}
        
        # Dequantization parameters from HEF
        # Scores: scale = 1/255, zp = 0
        self.score_scale = 0.003921568859368563
        self.score_zp = 0.0
        
        # Bbox dequant params per stride (from HEF inspection)
        # conv43 (stride 8): scale=0.0198, zp=4
        # conv50 (stride 16): scale=0.0318, zp=0
        # conv56 (stride 32): scale=0.0299, zp=1
        self.bbox_params = [
            (0.01984906569123268, 4.0),   # stride 8
            (0.031789254397153854, 0.0),  # stride 16
            (0.029924657195806503, 1.0),  # stride 32
        ]
        
        # Keypoints dequant params
        # conv44 (stride 8): scale=0.0261, zp=121
        # conv51 (stride 16): scale=0.0369, zp=115
        # conv57 (stride 32): scale=0.0360, zp=112
        self.kps_params = [
            (0.026139825582504272, 121.0),  # stride 8
            (0.036930251866579056, 115.0),  # stride 16
            (0.036007557064294815, 112.0),  # stride 32
        ]

    def dequantize(self, data, scale, zp):
        """Dequantize uint8 data to float."""
        return (data.astype(np.float32) - zp) * scale

    def postprocess(self, scores_list, bboxes_list, kpss_list, scale=1.0):
        pred_scores = []
        pred_bboxes = []
        pred_kpss = []

        if len(scores_list) != 3 or len(bboxes_list) != 3:
            return np.array([]), np.array([]), np.array([])

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores_raw = scores_list[idx][0]
            bbox_raw = bboxes_list[idx][0]
            kps_raw = kpss_list[idx][0]
            
            height, width = scores_raw.shape[:2]
            
            # Dequantize scores
            scores_dequant = self.dequantize(scores_raw, self.score_scale, self.score_zp)
            
            # Dequantize bbox and kps
            bbox_scale, bbox_zp = self.bbox_params[idx]
            kps_scale, kps_zp = self.kps_params[idx]
            
            bbox_dequant = self.dequantize(bbox_raw, bbox_scale, bbox_zp)
            kps_dequant = self.dequantize(kps_raw, kps_scale, kps_zp)
            
            # Build anchor centers
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
                anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers + 0.5) * stride
                anchor_centers = anchor_centers.reshape((-1, 2))
                anchor_centers = np.tile(anchor_centers, (1, self._num_anchors)).reshape((-1, 2))
                self.center_cache[key] = anchor_centers
            
            # Reshape to (H*W*num_anchors, C)
            scores = scores_dequant.reshape(-1)
            bbox_preds = bbox_dequant.reshape(-1, 4)
            kps_preds = kps_dequant.reshape(-1, 10)
            
            # Filter by confidence
            pos_inds = np.where(scores >= self.conf_thresh)[0]
            
            if len(pos_inds) == 0:
                continue
            
            # Decode bboxes - multiply by stride for distance
            bboxes = distance2bbox(anchor_centers, bbox_preds * stride)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            
            pred_scores.append(pos_scores)
            pred_bboxes.append(pos_bboxes)
            
            # Decode keypoints
            kpss = distance2kps(anchor_centers, kps_preds * stride)
            pos_kpss = kpss[pos_inds]
            pred_kpss.append(pos_kpss)

        if not pred_scores:
            return np.array([]), np.array([]), np.array([])

        scores = np.concatenate(pred_scores, axis=0)
        bboxes = np.concatenate(pred_bboxes, axis=0)
        kpss = np.concatenate(pred_kpss, axis=0) if pred_kpss else None

        if len(scores) > 0:
            keep = self.nms(bboxes, scores, self.nms_thresh)
            bboxes = bboxes[keep]
            scores = scores[keep]
            if kpss is not None:
                kpss = kpss[keep]

        # Rescale to original image size
        bboxes /= scale
        if kpss is not None:
            kpss /= scale

        return bboxes, kpss, scores

    def nms(self, boxes, scores, thresh):
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
