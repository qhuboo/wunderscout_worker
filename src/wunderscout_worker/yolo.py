import os
import time
import json
import random
from .lib.redis import redis_client

CLASSES = ["player", "ball", "referee"]

def fake_yolo_detection(frame_num):
    """Simulate YOLO detections for a frame"""
    detections = []
    num_objects = random.randint(1, 3)
    for _ in range(num_objects):
        detections.append({
            "class": random.choice(CLASSES),
            "confidence": round(random.uniform(0.7, 0.99), 2),
            "bbox": [
                random.randint(0, 640),  # x1
                random.randint(0, 480),  # y1
                random.randint(100, 640), # x2
                random.randint(100, 480)  # y2
            ]
        })
    return detections

def process_video(job_id, total_frames):
    print("WORKER: Processing video...")
    for frame in range(1, total_frames + 1):
        time.sleep(1)

        detections = fake_yolo_detection(frame)

        update = {
            "jobId": job_id,
            "status": "processing",
            "frame": frame,
            "detections": detections
        }
        print(f"update: {update}")
        redis_client.publish("job_updates", json.dumps(update))
        print(f"Published update: {update}")

    # Final completion message
    results = {
        "jobId": job_id,
        "status": "completed",
        "outputKey": f"results/{job_id}_processed.mp4"
    }
    redis_client.publish("job_updates", json.dumps(results))
    print(f"Published final results: {results}")