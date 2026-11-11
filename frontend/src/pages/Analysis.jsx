import { useState } from "react";
import { AlertCircle, ArrowRight, Clapperboard, Play, RotateCcw, ScanLine } from "lucide-react";
import AnalysisProgress from "../components/AnalysisProgress";
import DetectionViewer from "../components/DetectionViewer";
import EventTimeline from "../components/EventTimeline";
import GeneratedVideo from "../components/GeneratedVideo";
import PageHeading from "../components/PageHeading";
import VideoPreview from "../components/VideoPreview";
import VideoUploader from "../components/VideoUploader";
import { useAnalysis } from "../hooks/useAnalysis";

export default function Analysis() {
  const [file, setFile] = useState(null);
  const [eventType, setEventType] = useState("Auto-detect");
  const analysis = useAnalysis();

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    try {
      await analysis.analyze(file);
    } catch {
      // The hook stores and displays the request error.
    }
  }

  const payload = analysis.result?.result || analysis.result || {};
  const busy = analysis.status === "uploading" || analysis.status === "processing";

  return <>
    <PageHeading eyebrow="VIDEO WORKSPACE" title="Analyse an event" description="Upload a recording and prepare it for intelligent event analysis." />
    <div className="analysis-grid">
      <section className="panel">
        <div className="panel-heading">
          <div><span className="step-label">STEP 01</span><h2>Upload your video</h2><p>Choose a recorded cultural event, presentation or stage performance.</p></div>
          <span className="panel-icon"><Clapperboard size={20} /></span>
        </div>
        <form onSubmit={submit}>
          <VideoUploader onFileSelected={setFile} disabled={busy} />
          <label className="field-label event-type-label">Event category <span className="optional">Optional</span>
            <select className="select-control" value={eventType} onChange={(event) => setEventType(event.target.value)}>
              <option>Auto-detect</option><option>Cultural programme</option><option>Dance performance</option><option>Singing performance</option><option>Presentation</option><option>Speech</option><option>Panel discussion</option><option>Prize distribution</option>
            </select>
          </label>
          <button className="primary-button full" disabled={!file || busy}><Play size={16} /> Start video analysis <ArrowRight size={17} /></button>
        </form>
        <AnalysisProgress status={analysis.status} progress={analysis.progress} />
        {analysis.error && <div className="error-panel"><AlertCircle size={18} /><span>{analysis.error} Make sure the Flask backend is running and exposes the expected endpoint.</span></div>}
        {analysis.status !== "idle" && <button className="text-button" onClick={analysis.reset}><RotateCcw size={15} /> Reset analysis</button>}
      </section>
      <section className="panel">
        <div className="panel-heading"><div><span className="step-label">VIDEO PREVIEW</span><h2>Selected recording</h2><p>Check the video before sending it for analysis.</p></div></div>
        <VideoPreview file={file} />
        {file && <div className="video-meta"><span>{file.name}</span><small>{(file.size / (1024 * 1024)).toFixed(1)} MB</small></div>}
      </section>
    </div>

    <section className="section-block">
      <div className="section-title-row">
        <div><h2>Analysis output</h2><p>Detection frames, event summary and generated video.</p></div>
        <span className="status-pill neutral">{analysis.status === "completed" ? "Results received" : "Waiting for analysis"}</span>
      </div>
      {payload.summary && <div className="panel event-summary">
        <div>
          <span className="step-label">DETECTED EVENT</span>
          <h2>{payload.event_type || "Event summary"}</h2>
          <p>{payload.summary}</p>
          {payload.narrative && <p>{payload.narrative}</p>}
          {payload.peak_person_count != null && <p><strong>Peak visible people in a sampled frame:</strong> {payload.peak_person_count}. This is not a unique attendee count.</p>}
          {payload.action_predictions?.length > 0 && <p><strong>Suggested visible actions:</strong> {payload.action_predictions.map((item) => `${item.action_name} (${Math.round(item.confidence * 100)}%)`).join(", ")}</p>}
          {payload.model_quality && <small>Classifier: {payload.model_quality}</small>}
        </div>
        {payload.confidence != null && <span className="summary-confidence">{Math.round(payload.confidence * 100)}% confidence</span>}
      </div>}
      <div className="output-grid">
        <div className="panel">
          <div className="panel-heading"><div><h3><ScanLine size={17} /> Object detections</h3><p>Bounding boxes on a sampled frame.</p></div></div>
          <DetectionViewer imageUrl={payload.annotated_frame_url} detections={payload.detections || []} />
        </div>
        <div className="panel">
          <div className="panel-heading"><div><h3><Clapperboard size={17} /> Activity timeline</h3><p>Predicted event category and timestamps.</p></div></div>
          <EventTimeline events={payload.events || []} />
        </div>
      </div>
      {payload.generated_video_url && <div className="panel generated-video-panel">
        <div className="panel-heading"><div><h3>Generated video</h3><p>Sampled frames with object detections and event label.</p></div></div>
        <GeneratedVideo videoUrl={payload.generated_video_url} />
      </div>}
    </section>
  </>;
}
