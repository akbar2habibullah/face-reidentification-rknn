import os
import cv2
import numpy as np
from typing import Tuple

from utils.helpers import distance2bbox, distance2kps

# Import RKNNS and ONNX conditionally
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

__all__ = ["SCRFD"]


class SCRFD:
    """
    SCRFD Face Detection supporting ONNX and RKNN
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4
    ) -> None:
        
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model_path = model_path
        self.is_rknn = model_path.endswith('.rknn')

        # SCRFD model params
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        
        # ONNX specific params (baked into RKNN model, so only used for ONNX)
        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}

        if self.is_rknn:
            if not RKNN_AVAILABLE:
                raise ImportError("Model is .rknn but rknnlite is not installed.")
            self._init_rknn()
        else:
            if not ONNX_AVAILABLE:
                raise ImportError("Model is .onnx but onnxruntime is not installed.")
            self._init_onnx()

    def _init_onnx(self):
        try:
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            raise

    def _init_rknn(self):
        try:
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError("Load RKNN model failed")
            
            # Init runtime (NPU Core 0)
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            if ret != 0:
                raise RuntimeError("Init RKNN runtime failed")
            print("Initialized SCRFD RKNN model")
        except Exception as e:
            print(f"Failed to load RKNN model: {e}")
            raise

    def forward(self, image, threshold):
        # image is a padded/resized image of size (640, 640, 3)
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height = image.shape[0]
        input_width = image.shape[1]

        # 1. Inference
        if self.is_rknn:
            # RKNN: Input is RGB, Uint8, NHWC [1, 640, 640, 3]
            # Image is already BGR (opencv), convert to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_batch = np.expand_dims(img_rgb, axis=0)
            
            # Run inference
            outputs = self.rknn.inference(inputs=[img_batch], data_format='nhwc')
        else:
            # ONNX: BlobFromImage handles Mean/Std/SwapRB/NCHW
            blob = cv2.dnn.blobFromImage(
                image,
                1.0 / self.std,
                (input_width, input_height),
                (self.mean, self.mean, self.mean),
                swapRB=True
            )
            outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        # 2. Post-processing (Decoding)
        # Assuming output order is [score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32]
        # This loop logic relies on the outputs being in a specific order. 
        # Usually RKNN preserves the ONNX export order.
        
        fmc = self.fmc # 3
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Extract outputs for this stride
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc]
            bbox_preds = bbox_preds * stride
            
            if self.use_kps:
                kps_preds = outputs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            # Generate anchors
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # Thresholding
            pos_inds = np.where(scores >= threshold)[0]
            
            # Decode boxes
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            # Decode keypoints
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
                
        return scores_list, bboxes_list, kpss_list

    def detect(self, image, max_num=0, metric="max"):
        # This part handles the aspect-ratio preserving resize (Letterbox)
        width, height = self.input_size

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

        # Pad to exactly 640x640 (or configured input size)
        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image

        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.conf_thres)

        # Standard NMS logic
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        
        # Scale back to original image size
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        det = pre_det[keep, :]
        
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
            
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - image_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (area - offset_dist_squared * 2.0)
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
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