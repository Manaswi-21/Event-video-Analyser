from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import textwrap
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path

import cv2
import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from sklearn.neural_network import MLPClassifier
from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
LOCAL_FRONTEND_ENV = BASE_DIR.parent / "frontend" / ".env.local"
if LOCAL_FRONTEND_ENV.is_file():
    load_dotenv(LOCAL_FRONTEND_ENV)

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR))
UPLOAD_DIR = DATA_DIR / "uploads"
GENERATED_DIR = DATA_DIR / "generated"
DATABASE_PATH = DATA_DIR / "eventlens.sqlite3"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)


class EventPipeline:
    EVENT_LABELS = [
        "presentation",
        "speech",
        "panel discussion",
        "dance performance",
        "singing performance",
        "prize distribution",
        "cultural programme",
        "unknown",
    ]

    def __init__(self):
        self.model = self._build_model()
        configured_model = os.environ.get("EVENT_MLP_MODEL_PATH")
        model_path = Path(configured_model) if configured_model else BASE_DIR / "event_classifier.joblib"
        self.model_quality = "trained" if model_path.is_file() else "demo"
        self.yolo_model = self._load_yolo_model()

    def _load_yolo_model(self):
        if YOLO is None:
            logging.warning("Ultralytics is unavailable; using heuristic classification only.")
            return None
        try:
            return YOLO(str(BASE_DIR / "yolov8n.pt"))
        except Exception as exc:  # pragma: no cover
            logging.warning("YOLO model could not be loaded: %s", exc)
            return None

    def _build_model(self):
        configured_model = os.environ.get("EVENT_MLP_MODEL_PATH")
        model_path = Path(configured_model) if configured_model else BASE_DIR / "event_classifier.joblib"
        if model_path.is_file():
            return joblib.load(model_path)
        if configured_model:
            raise FileNotFoundError(f"Configured MLP model does not exist: {model_path}")

        examples = [
            [2, 1, 1, 0, 0, 0, 0.82],
            [1, 0, 0, 1, 0, 0, 0.86],
            [3, 1, 1, 1, 2, 0, 0.79],
            [4, 0, 0, 0, 0, 0, 0.71],
            [2, 0, 0, 1, 0, 0, 0.83],
            [2, 0, 0, 0, 0, 1, 0.74],
            [4, 1, 1, 0, 1, 0, 0.80],
            [0, 0, 0, 0, 0, 0, 0.10],
        ]
        labels = [
            "presentation",
            "speech",
            "panel discussion",
            "dance performance",
            "singing performance",
            "prize distribution",
            "cultural programme",
            "unknown",
        ]
        model = MLPClassifier(
            hidden_layer_sizes=(12, 8),
            max_iter=2000,
            random_state=42,
            learning_rate_init=0.02,
        )
        model.fit(np.array(examples, dtype=float), np.array(labels))
        return model

    def _feature_vector(self, detections, counts=None):
        counts = counts or Counter()
        confidence_values = []

        for detection in detections:
            label = str(detection.get("class_name", "unknown")).lower()
            confidence_values.append(float(detection.get("confidence", 0.0)))

        return [
            counts.get("person", 0),
            counts.get("laptop", 0),
            counts.get("tvmonitor", 0) + counts.get("monitor", 0),
            counts.get("microphone", 0),
            counts.get("chair", 0),
            counts.get("cell phone", 0) + counts.get("cellphone", 0),
            float(np.mean(confidence_values)) if confidence_values else 0.0,
        ]

    def _fallback_event_label(self, counts):
        people = counts.get("person", 0)
        laptop = counts.get("laptop", 0)
        microphone = counts.get("microphone", 0)
        chairs = counts.get("chair", 0)
        phone = counts.get("cell phone", 0) + counts.get("cellphone", 0)

        if people >= 1 and (laptop > 0 or counts.get("tvmonitor", 0) > 0):
            return "presentation"
        if microphone > 0 and people >= 1:
            return "speech"
        if people >= 3 and (chairs > 0 or microphone > 0):
            return "panel discussion"
        if people >= 2 and microphone > 0:
            return "singing performance"
        if people >= 2 and phone > 0:
            return "prize distribution"
        if people >= 2:
            return "cultural programme"
        return "unknown"

    def process_detections(self, detections, frame_detections=None):
        if not detections:
            return {
                "event_type": "unknown",
                "confidence": 0.0,
                "summary": "No clear event objects were detected in the uploaded video.",
                "top_objects": [],
                "peak_person_count": 0,
                "model_quality": self.model_quality,
            }

        frame_counts = []
        if frame_detections:
            for frame in frame_detections:
                frame_counts.append(Counter(
                    str(item.get("class_name", "unknown")).lower() for item in frame
                ))
        else:
            frame_counts.append(Counter(
                str(item.get("class_name", "unknown")).lower() for item in detections
            ))

        labels = set().union(*(frame_count.keys() for frame_count in frame_counts))
        counts = Counter({
            label: max(frame_count.get(label, 0) for frame_count in frame_counts)
            for label in labels
        })

        feature_vector = self._feature_vector(detections, counts)
        if self.model is not None:
            event_type = str(self.model.predict([feature_vector])[0])
            probabilities = self.model.predict_proba([feature_vector])[0]
            confidence = float(np.max(probabilities))
            supported_labels = getattr(self.model, "classes_", self.EVENT_LABELS)
        else:
            event_type = self._fallback_event_label(counts)
            confidence = 0.52
            supported_labels = self.EVENT_LABELS

        if event_type not in {str(label) for label in supported_labels}:
            event_type = "unknown"

        event_summary = self._build_summary(event_type, counts)
        return {
            "event_type": event_type,
            "confidence": round(float(confidence), 3),
            "summary": event_summary,
            "top_objects": [{"class_name": label, "count": count} for label, count in counts.most_common(4)],
            "peak_person_count": counts.get("person", 0),
            "model_quality": self.model_quality,
        }

    def _build_summary(self, event_type, counts):
        people = counts.get("person", 0)
        laptops = counts.get("laptop", 0)
        monitors = counts.get("tvmonitor", 0) + counts.get("monitor", 0)
        microphones = counts.get("microphone", 0)
        chairs = counts.get("chair", 0)

        if event_type == "unknown":
            return (
                f"The scene could not be matched confidently to a trained event category. "
                f"The peak number of visible person detections in a sampled frame was {people}."
            )

        observed_objects = ", ".join(
            f"{count} {name}(s)" for name, count in counts.most_common(4) if name != "person"
        )
        evidence = f" Other visible objects included {observed_objects}." if observed_objects else ""
        return (
            f'The video most closely matches the trained event category "{event_type}". '
            f"The peak number of visible person detections in a sampled frame was {people}."
            f"{evidence} The person count is not a unique attendee count."
        )

    def run_yolo(self, frame):
        if self.yolo_model is None:
            return []

        result = self.yolo_model(frame, conf=0.25, verbose=False)
        detections = []
        height, width = frame.shape[:2]

        for item in result:
            boxes = item.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls_index = int(box.cls[0].cpu().item())
                class_name = item.names.get(cls_index, "object")
                confidence = float(box.conf[0].cpu().item())
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [
                            round((x1 / width) * 100, 2),
                            round((y1 / height) * 100, 2),
                            round((x2 / width) * 100, 2),
                            round((y2 / height) * 100, 2),
                        ],
                    }
                )

        return detections


class VideoActionRecognizer:
    def __init__(self):
        self.enabled = os.environ.get("EVENT_ACTION_MODEL_ENABLED", "true").lower() not in {
            "0", "false", "no", "off"
        }
        self.model = None
        self.weights = None
        self.torch = None
        self.load_attempted = False
        self.status = "not_loaded" if self.enabled else "disabled"

    def _load_model(self):
        import torch
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=self.weights).eval()
        self.torch = torch
        self.status = "ready"

    def predict(self, frames):
        if not self.enabled or not frames:
            return []

        try:
            if self.model is None and self.load_attempted:
                return []
            if self.model is None:
                self.load_attempted = True
                self._load_model()

            indices = np.linspace(0, len(frames) - 1, num=16).round().astype(int)
            rgb_frames = [cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB) for index in indices]
            clip = self.torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2)
            clip = self.weights.transforms()(clip).unsqueeze(0)

            with self.torch.inference_mode():
                probabilities = self.model(clip)[0].softmax(dim=0)
                scores, class_indices = probabilities.topk(3)

            categories = self.weights.meta["categories"]
            self.status = "ready"
            return [
                {
                    "action_name": categories[index],
                    "confidence": round(float(score), 3),
                }
                for score, index in zip(scores.tolist(), class_indices.tolist())
            ]
        except Exception:
            self.status = "unavailable"
            logging.exception("Video action recognition could not be loaded or run")
            return []


def extract_sample_frames(video_path: str, max_frames: int = 16):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to read video file.")

    frames = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0 or len(frames) < max_frames:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        index += 1

    capture.release()
    return frames


def annotate_frame(frame, detections, label):
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        pt1 = (int((x1 / 100) * width), int((y1 / 100) * height))
        pt2 = (int((x2 / 100) * width), int((y2 / 100) * height))
        cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{detection['class_name']} {detection['confidence']:.2f}",
            (pt1[0], max(10, pt1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        label.upper(),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def create_summary_video(frames, label, output_path: Path, detections_by_frame=None):
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (width, height),
    )

    if not writer.isOpened():
        return None

    for index, frame in enumerate(frames):
        frame_detections = detections_by_frame[index] if detections_by_frame else []
        writer.write(annotate_frame(frame, frame_detections, label))
    writer.release()
    return output_path


def build_pdf_report(report: dict) -> bytes:
    def pdf_text(value):
        return value.encode("latin-1", errors="replace").decode("latin-1").replace(
            "\\", "\\\\"
        ).replace("(", "\\(").replace(")", "\\)")

    title = report.get("report_title") or f'{report.get("event_type", "Event").title()} report'
    narrative = report.get("narrative", report.get("summary", "Event summary"))
    lines = textwrap.wrap(narrative, width=88) or ["No narrative was generated."]
    commands = [
        "BT /F1 16 Tf 50 790 Td ({}) Tj".format(pdf_text(title)),
        "/F1 10 Tf 0 -28 Td",
    ]
    for line in lines[:45]:
        commands.append("({}) Tj 0 -14 Td".format(pdf_text(line)))
    commands.append("ET")
    stream = ("\n".join(commands) + "\n").encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def generate_report_narrative(event_result, detections, frame_count, action_predictions=None):
    observed_objects = event_result["top_objects"]
    if observed_objects:
        object_summary = ", ".join(
            f'{item["count"]} {item["class_name"]}(s)' for item in observed_objects
        )
        observation = f"The highest per-frame object counts were {object_summary}."
    else:
        observation = "No supported objects were detected in the sampled frames."

    confidence = event_result["confidence"]
    classifier_name = "trained event classifier" if event_result.get("model_quality") == "trained" else "demo event classifier"
    confidence_text = (
        f"The {classifier_name} returned a confidence score of {confidence:.0%}."
        if detections
        else "There was not enough visual evidence to assign an event category."
    )
    if action_predictions:
        suggested_actions = ", ".join(
            f'"{item["action_name"]}" ({item["confidence"]:.0%})'
            for item in action_predictions
        )
        action_text = (
            f" A pretrained video action model suggested {suggested_actions}. "
            "These are general action-class predictions, not verified descriptions of a person's job."
        )
    else:
        action_text = ""
    return (
        f'{event_result["summary"]} The analysis sampled {frame_count} frame(s). '
        f"{observation} {confidence_text}{action_text} This is an automated estimate based on sampled "
        "video frames and detected objects. The visible-person figure is a peak per-frame "
        "count, not a count of unique attendees. Review the original recording before "
        "relying on the predicted event category."
    )


pipeline = EventPipeline()
action_recognizer = VideoActionRecognizer()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", 250 * 1024 * 1024))
CORS(app, resources={r"/api/*": {"origins": "*"}})


@contextmanager
def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS reports_user_created ON reports(user_id, created_at DESC)"
        )


def save_report(report, user_id):
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO reports (id, user_id, created_at, report_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                created_at = excluded.created_at,
                report_json = excluded.report_json
            WHERE reports.user_id = excluded.user_id
            """,
            (report["id"], user_id, report["created_at"], json.dumps(report)),
        )


def get_user_reports(user_id):
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT report_json FROM reports WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [json.loads(row["report_json"]) for row in rows]


def get_user_report(user_id, report_id):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT report_json FROM reports WHERE user_id = ? AND id = ?",
            (user_id, report_id),
        ).fetchone()
    return json.loads(row["report_json"]) if row else None


def require_auth(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        project_url = (
            os.environ.get("SUPABASE_URL")
            or os.environ.get("VITE_SUPABASE_URL", "")
        ).rstrip("/")
        publishable_key = (
            os.environ.get("SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("VITE_SUPABASE_PUBLISHABLE_KEY", "")
        )
        authorization = request.headers.get("Authorization", "")
        if not project_url or not publishable_key:
            return jsonify({"error": "The API authentication service is not configured."}), 503
        if not authorization.startswith("Bearer "):
            return jsonify({"error": "Authentication is required."}), 401

        try:
            response = requests.get(
                f"{project_url}/auth/v1/user",
                headers={
                    "apikey": publishable_key,
                    "Authorization": authorization,
                },
                timeout=10,
            )
        except requests.RequestException:
            logging.exception("Supabase authentication request failed")
            return jsonify({"error": "Authentication service is temporarily unavailable."}), 502

        if response.status_code != 200:
            return jsonify({"error": "The access token is invalid or expired."}), 401

        try:
            user = response.json()
        except ValueError:
            logging.error("Supabase returned an invalid user response")
            return jsonify({"error": "Authentication service returned an invalid response."}), 502

        user_id = user.get("id") if isinstance(user, dict) else None
        if not user_id:
            logging.error("Supabase user response did not contain a user ID")
            return jsonify({"error": "Authentication service returned an invalid response."}), 502

        g.user_id = user_id
        return handler(*args, **kwargs)

    return wrapped


initialize_database()


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "The video exceeds the configured upload size limit."}), 413


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/media/<path:filename>", methods=["GET"])
@require_auth
def serve_media(filename):
    report_media = any(
        f"/api/media/{filename}" in (report.get("annotated_frame_url") or "")
        or f"/api/media/{filename}" in (report.get("generated_video_url") or "")
        for report in get_user_reports(g.user_id)
    )
    if not report_media:
        return jsonify({"error": "Media not found."}), 404
    return send_from_directory(GENERATED_DIR, filename)


@app.route("/api/analyze", methods=["POST"])
@require_auth
def analyze_video_route():
    if "video" not in request.files:
        return jsonify({"error": "No video file was included in the request."}), 400

    uploaded = request.files["video"]
    if uploaded.filename == "":
        return jsonify({"error": "No file was selected for analysis."}), 400

    safe_name = secure_filename(uploaded.filename)
    if not safe_name:
        return jsonify({"error": "The uploaded file name is invalid."}), 400
    saved_name = f"{uuid.uuid4()}_{safe_name}"
    video_path = UPLOAD_DIR / saved_name
    uploaded.save(video_path)

    try:
        frames = extract_sample_frames(str(video_path), max_frames=16)
        frame_detections = [pipeline.run_yolo(frame) for frame in frames]
        detections = [item for frame_items in frame_detections for item in frame_items]
        event_result = pipeline.process_detections(detections, frame_detections)
        action_predictions = action_recognizer.predict(frames)
        annotated_frames = [
            annotate_frame(frame, frame_items, event_result["event_type"])
            for frame, frame_items in zip(frames, frame_detections)
        ]

        narrative = generate_report_narrative(
            event_result, detections, len(frames), action_predictions
        )
        annotated_name = None
        if annotated_frames:
            annotated_name = f"{uuid.uuid4()}.jpg"
            annotated_path = GENERATED_DIR / annotated_name
            cv2.imwrite(str(annotated_path), annotated_frames[0])

        summary_path = GENERATED_DIR / f"{uuid.uuid4()}.mp4"
        generated_video = create_summary_video(
            frames, event_result["event_type"], summary_path, frame_detections
        )
        generated_url = f"/api/media/{summary_path.name}" if generated_video else None

        report = {
            "id": str(uuid.uuid4()),
            "report_title": f'{event_result["event_type"].title()} event report',
            "video_name": safe_name,
            "event_type": event_result["event_type"],
            "summary": event_result["summary"],
            "narrative": narrative,
            "confidence": event_result["confidence"],
            "model_quality": event_result["model_quality"],
            "peak_person_count": event_result["peak_person_count"],
            "sampled_frame_count": len(frames),
            "sampled_detection_count": len(detections),
            "detections": frame_detections[0] if frame_detections else [],
            "top_objects": event_result["top_objects"],
            "action_predictions": action_predictions,
            "action_model_status": action_recognizer.status,
            "events": [{
                "activity_name": event_result["event_type"],
                "start_seconds": 0,
                "end_seconds": max(1, len(frames) * 3),
                "confidence": event_result["confidence"],
            }],
            "annotated_frame_url": f"/api/media/{annotated_name}" if annotated_name else None,
            "generated_video_url": generated_url,
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }
        save_report(report, g.user_id)

        return jsonify({
            "result": {
                "annotated_frame_url": report["annotated_frame_url"],
                "detections": report["detections"],
                "events": report["events"],
                "summary": report["summary"],
                "narrative": narrative,
                "event_type": report["event_type"],
                "confidence": report["confidence"],
                "model_quality": report["model_quality"],
                "peak_person_count": report["peak_person_count"],
                "sampled_frame_count": report["sampled_frame_count"],
                "sampled_detection_count": report["sampled_detection_count"],
                "top_objects": report["top_objects"],
                "action_predictions": report["action_predictions"],
                "action_model_status": report["action_model_status"],
                "generated_video_url": generated_url,
            },
            "report_id": report["id"],
            "report": report,
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        logging.exception("Video analysis failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500
    finally:
        video_path.unlink(missing_ok=True)


@app.route("/api/reports", methods=["GET"])
@require_auth
def list_reports():
    return jsonify(get_user_reports(g.user_id))


@app.route("/api/reports/<report_id>", methods=["GET"])
@require_auth
def get_report(report_id):
    report = get_user_report(g.user_id, report_id)
    if report:
        return jsonify(report)
    return jsonify({"error": "Report not found"}), 404


@app.route("/api/reports/<report_id>/pdf", methods=["GET"])
@require_auth
def get_report_pdf(report_id):
    report = get_user_report(g.user_id, report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    return send_file(
        io.BytesIO(build_pdf_report(report)),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{report_id}.pdf",
    )


@app.route("/api/analyses", methods=["POST"])
@require_auth
def create_analysis_entry():
    payload = request.get_json(silent=True) or {}
    analysis_data = {
        "id": payload.get("id", str(uuid.uuid4())),
        "video_name": payload.get("video_name", "video.mp4"),
        "created_at": payload.get("created_at", datetime.utcnow().isoformat()),
        "status": payload.get("status", "completed"),
        "event_type": payload.get("event_type", "unknown"),
        "summary": payload.get("summary", "Analysis completed."),
    }
    save_report(analysis_data, g.user_id)
    return jsonify(analysis_data)


@app.route("/api/analyses/<analysis_id>", methods=["GET"])
@require_auth
def get_analysis_entry(analysis_id):
    record = get_user_report(g.user_id, analysis_id)
    if record:
        return jsonify(record)
    return jsonify({"error": "Analysis not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
