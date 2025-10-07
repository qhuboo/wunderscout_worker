import os
import threading
import time
from dotenv import load_dotenv
from ultralytics import YOLO
import torch
import supervision as sv
import cv2
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np
import umap
from sklearn.cluster import KMeans

load_dotenv()

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"

_EMBEDDINGS_MODEL = None
_EMBEDDINGS_PROCESSOR = None


def get_embeddings_model():
    global _EMBEDDINGS_MODEL, _EMBEDDINGS_PROCESSOR
    if _EMBEDDINGS_MODEL is None or _EMBEDDINGS_PROCESSOR is None:
        _EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(
            DEVICE
        )
        _EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    return _EMBEDDINGS_MODEL, _EMBEDDINGS_PROCESSOR


def create_player_crops(video_path):
    print("Generating player crops ...")

    # Getting training data for cluster model
    PLAYER_ID = 2
    STRIDE = 30

    model_trained = YOLO("/app/src/wunderscout_worker/best.pt")
    frame_generator = sv.get_video_frames_generator(
        source_path=video_path, stride=STRIDE
    )

    crops = []
    for i, frame in enumerate(frame_generator):
        print(f"WORKER: Processing frame {i}")
        result = model_trained.predict(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    print(f"Total crops collected: {len(crops)}.")

    # Convert to PIL
    pil_crops = [sv.cv2_to_pillow(c) for c in crops]
    return pil_crops


def extract_embeddings(crops):
    print("Extracting embeddings ...")
    BATCH_SIZE = 32
    model, processor = get_embeddings_model()

    batches = chunked(crops, BATCH_SIZE)
    data = []

    with torch.no_grad():
        for batch in batches:
            inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)

    data = np.concatenate(data)
    return data


def resolve_goalkeepers_team_id(players, goalkeepers):
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []

    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)


def yolo_detection(local_path, output_path):
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    CLASS_NAMES = {
        BALL_ID: "Ball",
        GOALKEEPER_ID: "Goalkeeper",
        PLAYER_ID: "Player",
        REFEREE_ID: "Referee",
    }

    video_path = local_path
    print(f"WORKER: This is the video_path: {video_path}")

    crops = create_player_crops(video_path)
    embeddings = extract_embeddings(crops)

    REDUCER = umap.UMAP(n_components=3)
    CLUSTERING_MODEL = KMeans(n_clusters=2, n_init=10, random_state=42)

    projections = REDUCER.fit_transform(embeddings)
    clustering_model = CLUSTERING_MODEL.fit(projections)

    model_trained = YOLO("/app/src/wunderscout_worker/best.pt")

    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)

    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    for i, frame in enumerate(frame_generator):
        print(f"WORKER: Processing frame {i}")
        result = model_trained.predict(frame, conf=0.3)[0]

        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[
            all_detections.class_id == GOALKEEPER_ID
        ]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        player_embeddings = extract_embeddings(players_crops)
        player_projection = REDUCER.transform(player_embeddings)
        players_detections.class_id = clustering_model.predict(player_projection)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections
        )

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge(
            [players_detections, goalkeepers_detections, referees_detections]
        )

        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        all_detections.class_id = all_detections.class_id.astype(int)

        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            thickness=2,
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1
        )

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame, detections=all_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=all_detections, labels=labels
        )
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame, detections=ball_detections
        )

        out.write(annotated_frame)
    out.release()


def process_video(local_path, output_path, job_id):
    print(
        f"[process_video][{time.strftime('%X')}] Job {job_id} running in thread {threading.get_ident()}"
    )
    if torch.cuda.is_available():
        print("GPU name: ", torch.cuda.get_device_name(0))
        yolo_detection(local_path, output_path)
    else:
        print("No GPU Access")
