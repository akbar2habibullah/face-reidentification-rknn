import cv2
import numpy as np
from logging import getLogger
import os

from utils.helpers import face_alignment

# Try to import RKNNLite, handle failure gracefully (e.g. if running on PC)
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

# Try to import ONNX Runtime
try:
    from onnxruntime import InferenceSession
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

__all__ = ["ArcFace"]

logger = getLogger(__name__)


class ArcFace:
    """
    ArcFace Model for Face Recognition supporting ONNX and RKNN (RK3566/RK3588)
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.input_size = (112, 112)
        # Constants for ONNX preprocessing
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5
        
        # Determine backend based on extension
        self.is_rknn = model_path.endswith('.rknn')

        logger.info(f"Initializing ArcFace model from {self.model_path}")

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
            self.session = InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            input_config = self.session.get_inputs()[0]
            self.input_name = input_config.name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            # Check input size
            input_shape = input_config.shape
            model_input_size = tuple(input_shape[2:4][::-1])
            if model_input_size != self.input_size:
                logger.warning(f"Model input size {model_input_size} differs from configured size {self.input_size}")
                
            self.embedding_size = self.session.get_outputs()[0].shape[1]
            logger.info(f"Initialized ONNX model. Embedding size: {self.embedding_size}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def _init_rknn(self):
        try:
            self.rknn = RKNNLite()
            
            # Load RKNN model
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError("Load RKNN model failed")
                
            # Init runtime
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError("Init RKNN runtime failed")
                
            logger.info("Initialized RKNN model successfully")
            
        except Exception as e:
            logger.error(f"Failed to load RKNN model: {e}")
            raise

    def preprocess_onnx(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess for ONNX: Resize -> Normalize (float) -> Transpose to NCHW
        """
        resized_face = cv2.resize(face_image, self.input_size)
        
        # Normalize
        face_blob = cv2.dnn.blobFromImage(
            resized_face,
            scalefactor=1.0 / self.normalization_scale,
            size=self.input_size,
            mean=(self.normalization_mean,)*3,
            swapRB=True # ArcFace standard is usually RGB
        )
        return face_blob

    def preprocess_rknn(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess for RKNN: Resize -> RGB -> Expand Dims
        Note: We do NOT normalize here because we baked mean/std into the RKNN model 
        during conversion. We send raw uint8 data.
        """
        resized_face = cv2.resize(face_image, self.input_size)
        
        # Convert BGR (OpenCV) to RGB
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        
        # RKNN Lite with data_format='nhwc' expects [1, H, W, C]
        face_blob = np.expand_dims(rgb_face, axis=0)
        return face_blob

    def get_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        normalized: bool = False
    ) -> np.ndarray:
        """
        Extract face embedding.
        """
        if image is None or landmarks is None:
            raise ValueError("Image and landmarks must not be None")

        try:
            # 1. Align the face
            aligned_face, _ = face_alignment(image, landmarks)

            # 2. Inference
            if self.is_rknn:
                # Preprocess (Raw RGB Uint8)
                face_blob = self.preprocess_rknn(aligned_face)
                
                # Inference
                # data_format='nhwc' is efficient for passing images to NPU
                outputs = self.rknn.inference(inputs=[face_blob], data_format='nhwc')
                embedding = outputs[0]
            else:
                # Preprocess (Normalized Float NCHW)
                face_blob = self.preprocess_onnx(aligned_face)
                outputs = self.session.run(self.output_names, {self.input_name: face_blob})
                embedding = outputs[0]

            # 3. Post-process (Flatten)
            embedding = embedding.flatten()

            # 4. Normalize (L2) - Crucial for Cosine Similarity
            if normalized or True: # ArcFace usually requires L2 norm
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    return embedding
                normalized_embedding = embedding / norm
                return normalized_embedding

        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            raise