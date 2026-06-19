import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Models and Source ---
MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.pt")
RTSP_URLS_STR = os.getenv("RTSP_URLS", "0")
RTSP_URLS = [int(url) if url.strip().isdigit() else url.strip() for url in RTSP_URLS_STR.split(',')]
LOG_PATH = os.getenv("LOG_PATH", "logs/detections.log")

# --- Detection Parameters ---
DETECT_CLASS = int(os.getenv("DETECT_CLASS", 0))
MIN_YOLO_CONFIDENCE = float(os.getenv("MIN_YOLO_CONFIDENCE", 0.75))
PERSON_NMS_IOU_THRESHOLD = float(os.getenv("PERSON_NMS_IOU_THRESHOLD", 0.45))

# --- Position Estimation Parameters ---
# These are approximate values for estimating depth from a single camera.
# Calibrate CAMERA_FOCAL_LENGTH_PX for better z/distance results.
PERSON_REAL_HEIGHT_M = float(os.getenv("PERSON_REAL_HEIGHT_M", 1.70))
CAMERA_FOCAL_LENGTH_PX = float(os.getenv("CAMERA_FOCAL_LENGTH_PX", 700.0))

# --- Tracking Parameters ---
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 0))
TRACKER_MAX_UNSEEN = int(os.getenv("TRACKER_MAX_UNSEEN", 10))
PERSON_TRACKER_MAX_DISTANCE = int(os.getenv("PERSON_TRACKER_MAX_DISTANCE", 75))
BOX_SMOOTHING_ALPHA = float(os.getenv("BOX_SMOOTHING_ALPHA", 0.70))
BOX_MAX_SIZE_CHANGE_RATIO = float(os.getenv("BOX_MAX_SIZE_CHANGE_RATIO", 0.20))
BOX_STATIONARY_CENTER_THRESHOLD = float(os.getenv("BOX_STATIONARY_CENTER_THRESHOLD", 8.0))

# --- State Management Parameters ---
STATE_FILE_PATH = os.getenv("STATE_FILE_PATH", "current_state.json")
HISTORY_FILE_PATH = os.getenv("HISTORY_FILE_PATH", "state_history.jsonl")
STATE_UPDATE_INTERVAL_SECONDS = int(os.getenv("STATE_UPDATE_INTERVAL_SECONDS", 5))
