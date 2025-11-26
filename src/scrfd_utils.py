import numpy as np
import cv2

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
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape((-1, 5, 2))

class SCRFDPostProc:
    def __init__(self, input_shape=(640, 640), conf_thresh=0.4, nms_thresh=0.4):
        self.input_shape = input_shape
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.center_cache = {}

    def postprocess(self, scores_list, bboxes_list, kpss_list, scale=1.0):
        """
        scores_list: list of [1, H*W*2, 1] 
        bboxes_list: list of [1, H*W*2, 4]
        kpss_list: list of [1, H*W*2, 10]
        """
        pred_scores = []
        pred_bboxes = []
        pred_kpss = []

        # Ensure lists are correct length
        if len(scores_list) != 3 or len(bboxes_list) != 3:
             # If logic fails or inputs missing, return empty
             return np.array([]), np.array([]), np.array([])

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = scores_list[idx][0] # (N, 1)
            bbox_preds = bboxes_list[idx][0] # (N, 4)
            kps_preds = kpss_list[idx][0] # (N, 10)
            
            height = self.input_shape[0] // stride
            width = self.input_shape[1] // stride
            
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.conf_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            
            pred_scores.append(pos_scores)
            pred_bboxes.append(pos_bboxes)
            
            if kps_preds is not None:
                kpss = distance2kps(anchor_centers, kps_preds)
                pos_kpss = kpss[pos_inds]
                pred_kpss.append(pos_kpss)

        if not pred_scores:
            return np.array([]), np.array([]), np.array([])

        scores = np.concatenate(pred_scores, axis=0)
        bboxes = np.concatenate(pred_bboxes, axis=0)
        kpss = np.concatenate(pred_kpss, axis=0) if pred_kpss else None

        # NMS
        keep = self.nms(bboxes, scores, self.nms_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        if kpss is not None:
            kpss = kpss[keep]

        # Rescale
        bboxes /= scale
        if kpss is not None:
            kpss /= scale

        return bboxes, kpss, scores

    def nms(self, boxes, scores, thresh):
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

