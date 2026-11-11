import io
from types import SimpleNamespace

import app as backend
from app import EventPipeline, build_pdf_report, generate_report_narrative
import numpy as np
from sklearn.dummy import DummyClassifier


def test_event_pipeline_classifies_event_and_summary():
    pipeline = EventPipeline()

    sample_detections = [
        {"class_name": "person", "confidence": 0.95, "bbox": [10, 20, 40, 80]},
        {"class_name": "person", "confidence": 0.94, "bbox": [50, 18, 76, 82]},
        {"class_name": "laptop", "confidence": 0.87, "bbox": [30, 40, 55, 70]},
        {"class_name": "tvmonitor", "confidence": 0.82, "bbox": [80, 30, 95, 70]},
    ]

    result = pipeline.process_detections(sample_detections)

    assert result["event_type"] in {
        "presentation",
        "speech",
        "panel discussion",
    }
    assert "summary" in result
    assert len(result["summary"]) > 20
    assert result["confidence"] >= 0.0
    assert result["confidence"] <= 1.0


def test_event_pipeline_handles_empty_detections():
    pipeline = EventPipeline()

    result = pipeline.process_detections([])

    assert result["event_type"] == "unknown"
    assert "summary" in result
    assert result["model_quality"] == "demo"


def test_event_pipeline_counts_peak_people_per_frame_instead_of_recounting_them():
    pipeline = EventPipeline()
    first_frame = [
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "person", "confidence": 0.8},
        {"class_name": "chair", "confidence": 0.9},
    ]
    second_frame = [
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "person", "confidence": 0.8},
        {"class_name": "chair", "confidence": 0.9},
    ]

    result = pipeline.process_detections(first_frame + second_frame, [first_frame, second_frame])

    assert result["peak_person_count"] == 2
    assert {item["class_name"]: item["count"] for item in result["top_objects"]}["person"] == 2


def test_event_pipeline_accepts_labels_from_trained_dataset():
    pipeline = EventPipeline()
    trained_classifier = DummyClassifier(strategy="constant", constant="town hall")
    trained_classifier.fit([[0], [1]], ["town hall", "town hall"])
    pipeline.model = trained_classifier

    result = pipeline.process_detections([
        {"class_name": "person", "confidence": 0.93},
    ])

    assert result["event_type"] == "town hall"
    assert '"town hall"' in result["summary"]


def test_report_narrative_includes_observations_and_prototype_disclaimer():
    narrative = generate_report_narrative(
        {
            "summary": "A presentation was detected.",
            "confidence": 0.75,
            "top_objects": [{"class_name": "person", "count": 2}],
        },
        [{"class_name": "person", "confidence": 0.9}],
        4,
    )

    assert "sampled 4 frame(s)" in narrative
    assert "2 person(s)" in narrative
    assert "automated estimate" in narrative


def test_report_narrative_includes_action_model_predictions_as_suggestions():
    narrative = generate_report_narrative(
        {
            "summary": "The video most closely matches a workshop.",
            "confidence": 0.8,
            "top_objects": [{"class_name": "person", "count": 2}],
            "peak_person_count": 2,
        },
        [{"class_name": "person", "confidence": 0.9}],
        16,
        [{"action_name": "typing", "confidence": 0.84}],
    )

    assert 'action model suggested "typing"' in narrative


def test_generated_report_pdf_has_valid_cross_reference_offsets():
    pdf = build_pdf_report({
        "report_title": "Event report",
        "narrative": "An event was detected (prototype).",
    })
    xref_offset = int(pdf.rsplit(b"startxref\n", 1)[1].splitlines()[0])

    assert pdf.startswith(b"%PDF-1.4")
    assert pdf[xref_offset:].startswith(b"xref\n")
    assert b"An event was detected \\(prototype\\)." in pdf


def test_health_endpoint_does_not_require_authentication():
    response = backend.app.test_client().get("/api/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_api_requires_supabase_configuration(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.delenv("VITE_SUPABASE_URL", raising=False)
    monkeypatch.delenv("VITE_SUPABASE_PUBLISHABLE_KEY", raising=False)

    response = backend.app.test_client().get("/api/reports")

    assert response.status_code == 503
    assert response.json["error"] == "The API authentication service is not configured."


def test_backend_accepts_frontend_supabase_variables_without_a_token(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.setenv("VITE_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("VITE_SUPABASE_PUBLISHABLE_KEY", "publishable-key")

    response = backend.app.test_client().get("/api/reports")

    assert response.status_code == 401
    assert response.json["error"] == "Authentication is required."


def test_reports_are_persisted_and_isolated_by_user(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DATABASE_PATH", tmp_path / "reports.sqlite3")
    backend.initialize_database()
    report = {
        "id": "report-1",
        "created_at": "2026-01-01T00:00:00",
        "summary": "A test report",
    }
    backend.save_report(report, "user-1")

    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "publishable-key")
    def authenticate(*args, **kwargs):
        user_id = kwargs["headers"]["Authorization"].removeprefix("Bearer ")
        return SimpleNamespace(status_code=200, json=lambda: {"id": user_id})

    monkeypatch.setattr(backend.requests, "get", authenticate)
    client = backend.app.test_client()

    own_report = client.get(
        "/api/reports/report-1",
        headers={"Authorization": "Bearer user-1"},
    )
    other_user_report = client.get(
        "/api/reports/report-1",
        headers={"Authorization": "Bearer user-2"},
    )

    assert own_report.status_code == 200
    assert own_report.json["summary"] == "A test report"
    assert other_user_report.status_code == 404


def test_analyze_route_returns_peak_people_count_and_runs_detection_once_per_frame(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(backend, "DATABASE_PATH", tmp_path / "reports.sqlite3")
    backend.initialize_database()
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "publishable-key")
    monkeypatch.setattr(
        backend.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"id": "user-1"},
        ),
    )
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(backend, "extract_sample_frames", lambda *args, **kwargs: frames)
    detected = [
        {"class_name": "person", "confidence": 0.95, "bbox": [5, 5, 30, 45]},
        {"class_name": "person", "confidence": 0.92, "bbox": [35, 5, 60, 45]},
        {"class_name": "chair", "confidence": 0.89, "bbox": [20, 25, 45, 45]},
    ]
    yolo_calls = []

    def detect(_frame):
        yolo_calls.append(True)
        return detected

    monkeypatch.setattr(backend.pipeline, "run_yolo", detect)
    monkeypatch.setattr(backend.action_recognizer, "predict", lambda _frames: [])
    monkeypatch.setattr(backend.cv2, "imwrite", lambda *args, **kwargs: True)
    monkeypatch.setattr(backend, "create_summary_video", lambda *args, **kwargs: None)

    response = backend.app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"test video"), "event.mp4")},
        headers={"Authorization": "Bearer test-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["result"]["peak_person_count"] == 2
    assert response.json["result"]["sampled_detection_count"] == 6
    assert "peak number of visible person detections" in response.json["result"]["narrative"]
    assert len(yolo_calls) == len(frames)
