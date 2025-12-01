import os
import cv2
import random
import time
import warnings
import argparse
import logging
import numpy as np

from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")
setup_logging(log_to_file=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition with FAISS")

    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_mbf.onnx", help="Path to recognition model")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")
    parser.add_argument("--faces-dir", type=str, default="./assets/faces", help="Path to faces stored dir")
    # Changed default source to '0' for webcam ease of use, though command line override is preferred
    parser.add_argument("--source", type=str, default="0", help="Video file or webcam index (e.g. 0)")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum number of face detections from a frame")
    parser.add_argument(
        "--db-path",
        type=str,
        default="./database/face_database",
        help="path to vector db and metadata"
    )
    parser.add_argument("--update-db", action="store_true", help="Force update of the face database")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Output path for annotated video")
    parser.add_argument("--no-display", action="store_true", help="Disable video display window (useful for headless)")

    return parser.parse_args()

def build_face_database(detector: SCRFD, recognizer: ArcFace, params: argparse.Namespace, force_update: bool = False) -> FaceDatabase:
    face_db = FaceDatabase(db_path=params.db_path, max_workers=4)

    if not force_update and face_db.load():
        logging.info("Loaded face database from disk.")
        return face_db

    logging.info("Building face database from images...")

    if not os.path.exists(params.faces_dir):
        logging.warning(f"Faces directory {params.faces_dir} does not exist. Creating empty database.")
        face_db.save()
        return face_db

    for filename in os.listdir(params.faces_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        name = filename.rsplit('.', 1)[0]
        image_path = os.path.join(params.faces_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"Could not read image: {image_path}")
            continue

        try:
            bboxes, kpss = detector.detect(image, max_num=1)

            if len(kpss) == 0:
                logging.warning(f"No face detected in {image_path}. Skipping...")
                continue

            embedding = recognizer.get_embedding(image, kpss[0])
            face_db.add_face(embedding, name)
            logging.info(f"Added face for: {name}")
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            continue

    face_db.save()
    return face_db

def frame_processor(frame: np.ndarray, detector: SCRFD, recognizer: ArcFace, face_db: FaceDatabase, colors: dict, params: argparse.Namespace) -> np.ndarray:
    try:
        bboxes, kpss = detector.detect(frame, params.max_num)

        if len(bboxes) == 0:
            return frame

        # Process all faces in the frame
        embeddings = []
        processed_bboxes = []

        # Get embeddings for all faces
        for bbox, kps in zip(bboxes, kpss):
            try:
                *bbox_coords, conf_score = bbox.astype(np.int32)
                embedding = recognizer.get_embedding(frame, kps)
                embeddings.append(embedding)
                processed_bboxes.append(bbox_coords)
            except Exception as e:
                logging.warning(f"Error processing face embedding: {e}")
                continue

        if not embeddings:
            return frame

        # Batch search for all faces
        results = face_db.batch_search(embeddings, params.similarity_thresh)

        # Draw results
        for bbox, (name, similarity) in zip(processed_bboxes, results):
            if name != "Unknown":
                if name not in colors:
                    colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=colors[name])
            else:
                draw_bbox(frame, bbox, (0, 0, 255)) # Red for unknown

    except Exception as e:
        logging.error(f"Error in frame processing: {e}")

    return frame

def draw_performance_stats(frame, fps, inference_time_ms):
    """Draw FPS and Inference time on the frame"""
    # Create a semi-transparent background for text
    h, w, _ = frame.shape
    
    stats_text = [
        f"FPS: {fps:.1f}",
        f"Latency: {inference_time_ms:.1f}ms"
    ]
    
    for i, text in enumerate(stats_text):
        y_pos = 30 + (i * 30)
        # Draw shadow/outline for better visibility
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        # Draw text
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def main(params):
    try:
        logging.info("Loading models...")
        detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
        recognizer = ArcFace(params.rec_weight)
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    # Use context manager for proper resource cleanup
    with build_face_database(detector, recognizer, params, force_update=params.update_db) as face_db:
        colors = {}

        try:
            # Source handling: if it looks like a number, convert to int for webcam
            source = params.source
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                logging.info(f"Opening Webcam Index: {source}")
            else:
                logging.info(f"Opening Video File: {source}")

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise IOError(f"Could not open video source: {source}")

            # Setup video writer
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = cap.get(cv2.CAP_PROP_FPS)
            # Default to 30 if camera doesn't report FPS
            if fps_input == 0: fps_input = 30.0 
            
            out = cv2.VideoWriter(params.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_input, (width, height))

            frame_count = 0
            
            # FPS Calculation variables
            prev_frame_time = 0
            new_frame_time = 0
            fps_avg = 0.0
            alpha = 0.1  # Smoothing factor for FPS (0.1 means slow smoothing, 0.9 means fast reaction)

            logging.info("Starting processing loop...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, int): 
                        logging.error("Failed to grab frame from camera.")
                        continue # Keep trying for webcam
                    else:
                        break # End of video file

                # Performance timing
                start_process = time.time()
                
                # Core Processing
                frame = frame_processor(frame, detector, recognizer, face_db, colors, params)
                
                end_process = time.time()
                inference_time = (end_process - start_process) * 1000 # in ms

                # FPS Calculation
                new_frame_time = time.time()
                # Protect against division by zero on very first frame or super fast processing
                time_diff = new_frame_time - prev_frame_time
                if time_diff > 0:
                    curr_fps = 1.0 / time_diff
                    # Exponential Moving Average for smoother FPS display
                    fps_avg = (1 - alpha) * fps_avg + alpha * curr_fps
                prev_frame_time = new_frame_time

                # Draw Stats
                draw_performance_stats(frame, fps_avg, inference_time)

                # Write to output file
                out.write(frame)

                # Display
                if not params.no_display:
                    cv2.imshow("Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if frame_count % 30 == 0:
                    logging.info(f"Frame {frame_count} | FPS: {fps_avg:.2f} | Latency: {inference_time:.2f}ms")

            logging.info(f"Processed {frame_count} frames.")

        except Exception as e:
            logging.error(f"Error during video processing: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)