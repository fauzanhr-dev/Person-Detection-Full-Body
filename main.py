import cv2
from ultralytics import YOLO
import numpy as np
import time
import logging
import threading
import queue

import config
import state_manager
from logger_config import setup_logging

GLOBAL_LOCK = threading.Lock()
save_queue = queue.Queue()
frame_queue = queue.Queue() 

def estimate_person_position(frame, bbox):
    """
    Estimates a person's image position and approximate camera-space position.
    A single RGB camera cannot measure true depth, so z is estimated from bbox height.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_width = max(x2 - x1, 1)
    bbox_height = max(y2 - y1, 1)
    center_x = x1 + bbox_width / 2
    center_y = y1 + bbox_height / 2
    z_m = (config.PERSON_REAL_HEIGHT_M * config.CAMERA_FOCAL_LENGTH_PX) / bbox_height
    x_m = ((center_x - frame_width / 2) * z_m) / config.CAMERA_FOCAL_LENGTH_PX
    y_m = ((center_y - frame_height / 2) * z_m) / config.CAMERA_FOCAL_LENGTH_PX

    return {
        "image": {
            "x": round(center_x, 2),
            "y": round(center_y, 2),
            "bbox_width": bbox_width,
            "bbox_height": bbox_height
        },
        "camera": {
            "x": round(x_m, 3),
            "y": round(y_m, 3),
            "z": round(z_m, 3)
        }
    }

def smooth_bbox(previous_bbox, current_bbox):
    alpha = min(max(config.BOX_SMOOTHING_ALPHA, 0.0), 1.0)
    return tuple(
        int(round(previous_value * alpha + current_value * (1 - alpha)))
        for previous_value, current_value in zip(previous_bbox, current_bbox)
    )

def bbox_iou(first_bbox, second_bbox):
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox
    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_width = max(inter_x2 - inter_x1, 0)
    inter_height = max(inter_y2 - inter_y1, 0)
    inter_area = inter_width * inter_height
    first_area = max(first_x2 - first_x1, 0) * max(first_y2 - first_y1, 0)
    second_area = max(second_x2 - second_x1, 0) * max(second_y2 - second_y1, 0)
    union_area = first_area + second_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def suppress_overlapping_detections(detections):
    kept_detections = []
    for detection in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        if all(
            bbox_iou(detection['bbox'], kept_detection['bbox']) <= config.PERSON_NMS_IOU_THRESHOLD
            for kept_detection in kept_detections
        ):
            kept_detections.append(detection)
    return kept_detections

def detection_area(detection):
    x1, y1, x2, y2 = detection['bbox']
    return max(x2 - x1, 0) * max(y2 - y1, 0)

def limit_bbox_size_change(previous_bbox, current_bbox):
    max_change_ratio = max(config.BOX_MAX_SIZE_CHANGE_RATIO, 0.0)
    if max_change_ratio == 0:
        return current_bbox

    prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox
    curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
    prev_width = max(prev_x2 - prev_x1, 1)
    prev_height = max(prev_y2 - prev_y1, 1)
    curr_width = max(curr_x2 - curr_x1, 1)
    curr_height = max(curr_y2 - curr_y1, 1)

    min_width = prev_width * (1 - max_change_ratio)
    max_width = prev_width * (1 + max_change_ratio)
    min_height = prev_height * (1 - max_change_ratio)
    max_height = prev_height * (1 + max_change_ratio)
    limited_width = min(max(curr_width, min_width), max_width)
    limited_height = min(max(curr_height, min_height), max_height)
    center_x = curr_x1 + curr_width / 2
    center_y = curr_y1 + curr_height / 2

    return (
        int(round(center_x - limited_width / 2)),
        int(round(center_y - limited_height / 2)),
        int(round(center_x + limited_width / 2)),
        int(round(center_y + limited_height / 2))
    )

def lock_stationary_bbox_size(previous_bbox, current_bbox):
    prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox
    curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
    prev_width = max(prev_x2 - prev_x1, 1)
    prev_height = max(prev_y2 - prev_y1, 1)
    curr_width = max(curr_x2 - curr_x1, 1)
    curr_height = max(curr_y2 - curr_y1, 1)
    prev_center = (prev_x1 + prev_width / 2, prev_y1 + prev_height / 2)
    curr_center = (curr_x1 + curr_width / 2, curr_y1 + curr_height / 2)
    center_distance = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])

    if center_distance > config.BOX_STATIONARY_CENTER_THRESHOLD:
        return current_bbox

    return (
        int(round(curr_center[0] - prev_width / 2)),
        int(round(curr_center[1] - prev_height / 2)),
        int(round(curr_center[0] + prev_width / 2)),
        int(round(curr_center[1] + prev_height / 2))
    )

def process_stream(stream_url, stream_id):
    """
    Processes a single video stream in a dedicated thread.
    Manages tracking.
    """
    logging.info(f"--- Starting processing for stream {stream_id} ({stream_url}) ---")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logging.error(f"[{stream_id}] Error: Unable to open stream: {stream_url}")
        return

    trackers = {}
    next_tracker_id = 0
    frame_counter = 0
    last_annotations = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"[{stream_id}] Frame not received, end of stream or error.")
            time.sleep(2)
            cap.open(stream_url)
            continue

        process_this_frame = config.FRAME_SKIP == 0 or (frame_counter + 1) % (config.FRAME_SKIP + 1) == 0
        if process_this_frame:
            last_annotations = []
        frame_counter += 1

        if process_this_frame:
            yolo_results = model(
                frame,
                conf=config.MIN_YOLO_CONFIDENCE,
                classes=[config.DETECT_CLASS],
                verbose=False
            )
            persons_in_frame = []
            matched_tracker_ids = set()
            for t_data in trackers.values():
                t_data['unseen_frames'] += 1

            detections = []
            for box in yolo_results[0].boxes:
                if int(box.cls) != config.DETECT_CLASS or float(box.conf) < config.MIN_YOLO_CONFIDENCE:
                    continue
                raw_bbox = tuple(map(int, box.xyxy[0].tolist()))
                x1_raw, y1_raw, x2_raw, y2_raw = raw_bbox
                person_crop = frame[y1_raw:y2_raw, x1_raw:x2_raw]
                if person_crop.size == 0: continue
                detections.append({
                    'bbox': raw_bbox,
                    'confidence': float(box.conf),
                    'class_id': int(box.cls)
                })

            filtered_detections = suppress_overlapping_detections(detections)
            filtered_detections.sort(key=detection_area, reverse=True)
            for detection in filtered_detections:
                raw_bbox = detection['bbox']
                raw_position = estimate_person_position(frame, raw_bbox)
                center = (raw_position['image']['x'], raw_position['image']['y'])
                best_tracker_id = None
                best_distance = float('inf')
                for tracker_id, t_data in trackers.items():
                    if tracker_id in matched_tracker_ids:
                        continue
                    distance = np.hypot(center[0] - t_data['center'][0], center[1] - t_data['center'][1])
                    if distance < best_distance and distance <= config.PERSON_TRACKER_MAX_DISTANCE:
                        best_distance = distance
                        best_tracker_id = tracker_id

                if best_tracker_id is None:
                    person_id = next_tracker_id
                    next_tracker_id += 1
                else:
                    person_id = best_tracker_id
                matched_tracker_ids.add(person_id)

                if person_id in trackers and 'bbox' in trackers[person_id]:
                    previous_bbox = trackers[person_id]['bbox']
                    stationary_bbox = lock_stationary_bbox_size(previous_bbox, raw_bbox)
                    limited_bbox = limit_bbox_size_change(previous_bbox, stationary_bbox)
                    display_bbox = smooth_bbox(previous_bbox, limited_bbox)
                else:
                    display_bbox = raw_bbox

                x1_person, y1_person, x2_person, y2_person = display_bbox
                position = estimate_person_position(frame, display_bbox)
                smoothed_center = (position['image']['x'], position['image']['y'])
                trackers[person_id] = {
                    'center': smoothed_center,
                    'bbox': display_bbox,
                    'unseen_frames': 0,
                    'status': 'Active'
                }

                confidence = detection['confidence']
                top_label = f"ID={person_id} person={confidence:.2f}"
                bottom_label = (
                    f"x={position['camera']['x']:.2f}m "
                    f"y={position['camera']['y']:.2f}m "
                    f"z={position['camera']['z']:.2f}m"
                )
                top_y = max(y1_person - 10, 20)
                bottom_y = min(y2_person + 22, frame.shape[0] - 10)
                last_annotations.append({'type': 'rect', 'args': [(x1_person, y1_person), (x2_person, y2_person), (255, 128, 0), 2]})
                last_annotations.append({'type': 'text', 'args': [top_label, (x1_person, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2]})
                last_annotations.append({'type': 'text', 'args': [bottom_label, (x1_person, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2]})
                persons_in_frame.append({
                    "person_id": person_id,
                    "class_id": detection['class_id'],
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": x1_person,
                        "y1": y1_person,
                        "x2": x2_person,
                        "y2": y2_person
                    },
                    "position": position
                })

            stale_trackers = []
            for tracker_id, t_data in trackers.items():
                if t_data['unseen_frames'] > config.TRACKER_MAX_UNSEEN or t_data.get('status') == 'Promoted':
                    stale_trackers.append(tracker_id)
            for tracker_id in stale_trackers:
                logging.info(f"[{stream_id}] Removing tracker {tracker_id} (Stale or Promoted).")
                del trackers[tracker_id]
            with global_states_lock:
                global_stream_states[stream_id] = persons_in_frame
        for ann in last_annotations:
            if ann['type'] == 'rect':
                cv2.rectangle(frame, *ann['args'])
            elif ann['type'] == 'text':
                cv2.putText(frame, *ann['args'])
            elif ann['type'] == 'circle':
                cv2.circle(frame, *ann['args'])
        try:
            frame_queue.put_nowait((stream_id, frame))
        except queue.Full:
            logging.warning(f"[{stream_id}] Frame queue is full, a frame has been dropped to avoid delay.")
    cap.release()
    logging.info(f"--- Processing finished for stream {stream_id} ---")

def main():
    """
    Main function: starts stream processing threads, I/O management, and display.
    Handles status updates and controlled shutdown.
    """
    global model, stop_event, global_stream_states, global_states_lock, last_state_update_time
    setup_logging()
    logging.info("Loading YOLO model...")
    model = YOLO(config.MODEL_PATH)
    stop_event = threading.Event()
    global_stream_states = {}
    global_states_lock = threading.Lock()
    threads = []
    logging.info(f"Starting {len(config.RTSP_URLS)} stream(s)...")
    for i, rtsp_url in enumerate(config.RTSP_URLS):
        thread_name = f"StreamThread-{i}"
        thread = threading.Thread(target=process_stream, args=(rtsp_url, i), name=thread_name)
        threads.append(thread)
        thread.start()
        logging.info(f"Thread '{thread_name}' started for stream URL: {rtsp_url}")
    last_state_update_time = time.time()
    try:
        while not stop_event.is_set():
            if not all(t.is_alive() for t in threads):
                logging.warning("One or more stream threads have stopped. Exiting.")
                stop_event.set()
                break
            try:
                stream_id, frame = frame_queue.get_nowait()
                window_name = f"Stream {stream_id}"
                cv2.imshow(window_name, frame)
            except queue.Empty:
                pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("'q' key pressed. Starting shutdown procedure...")
                stop_event.set()
                break
            now = time.time()
            if now - last_state_update_time >= config.STATE_UPDATE_INTERVAL_SECONDS:
                with global_states_lock:
                    current_states_copy = dict(global_stream_states)
                state_manager.update_state(current_states_copy)
                last_state_update_time = now
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Starting shutdown procedure...")
        stop_event.set()
    finally:
        logging.info("Waiting for all threads to terminate...")
        for thread in threads:
            thread.join()
        logging.info("All threads terminated. Clean exit.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
