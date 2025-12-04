import numpy as np
import cv2
import logging
from ..utils import distance2bbox, distance2kps

# Runtime checks
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger("scrfd")

class SCRFD:
    def __init__(self, model_path, input_size=(640, 640), conf_thres=0.5, iou_thres=0.4):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model_path = model_path
        
        # SCRFD constants
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.center_cache = {}
        
        # Init Runtime
        self.is_rknn = model_path.endswith('.rknn')
        if self.is_rknn:
            self._init_rknn()
        else:
            self._init_onnx()

    def _init_rknn(self):
        logger.info(f"Loading SCRFD RKNN: {self.model_path}")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0: raise Exception("Load RKNN failed")
        if self.rknn.init_runtime() != 0: raise Exception("Init RKNN failed")

    def _init_onnx(self):
        logger.info(f"Loading SCRFD ONNX: {self.model_path}")
        self.session = onnxruntime.InferenceSession(self.model_path)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

    def forward(self, image, threshold):
        input_height, input_width = image.shape[:2]
        
        # 1. Inference
        if self.is_rknn:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_batch = np.expand_dims(img_rgb, axis=0)
            outputs = self.rknn.inference(inputs=[img_batch], data_format='nhwc')
        else:
            # ONNX preprocessing
            blob = cv2.dnn.blobFromImage(image, 1.0/128.0, (input_width, input_height), (127.5, 127.5, 127.5), swapRB=True)
            outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        # 2. Decode
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride
            
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                y, x = np.mgrid[:height, :width]
                anchor_centers = np.stack([x, y], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def detect(self, image, max_num=0, metric="max"):
        width, height = self.input_size

        # --- EXACT REFERENCE LOGIC: Aspect Ratio Preserving Resize ---
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))

        # Pad to 640x640
        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image
        # -----------------------------------------------------------

        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.conf_thres)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        # Rescale bboxes/kpss back to original image size
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        
        det = pre_det[keep, :]
        kpss = kpss[order, :, :][keep, :, :]

        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0],
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (area - offset_dist_squared * 2.0)
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            kpss = kpss[bindex, :, :]

        return det, kpss

    def nms(self, dets, iou_thres):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
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
            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]
        return keep