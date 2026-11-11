# Event classifier training data

The API can load a trained MLP artifact through `EVENT_MLP_MODEL_PATH`. If the variable is unset, it uses the demo classifier in `app.py`.

## CSV format

Create one row per labeled video. Counts should describe the maximum number of detections for each object class in any one sampled frame, not the sum across frames. This matches the API's peak-per-frame feature extraction.

Required columns:

```csv
persons,laptops,monitors,microphones,chairs,phones,mean_confidence,event_type
```

`event_type` can be any label represented in your labeled dataset, for example `lecture` or `workshop`. Use enough correctly labeled videos per event category for the MLP to generalize; a few hand-authored rows are only a pipeline smoke test, not a reliable model.

Train and save the artifact:

```powershell
python train_model.py path\to\labeled-videos.csv --output event_classifier.joblib
```

Set `EVENT_MLP_MODEL_PATH` to the artifact's path when starting the API. Deploy the same artifact alongside the backend, or use a persistent model storage location.

## Model scope

The default YOLOv8n COCO model detects general objects and people. It does not identify a person's age or gender. The API also uses a pretrained R3D-18 Kinetics-400 video model to suggest general visible actions from a 16-frame clip. Its weights load lazily on the first analysis and require internet access if they are not cached. Set `EVENT_ACTION_MODEL_ENABLED=false` to disable action inference; event detection and reporting continue to work.

Action labels describe general movements, not a person's occupation or intent. The report presents them as model suggestions. The people figure is the peak visible count in a sampled frame, not unique attendees. Age/gender are not inferred from appearance; if demographics are required, use voluntary, consented, self-reported data rather than visual guesses.
