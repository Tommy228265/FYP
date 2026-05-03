## Functional Scope

### Real-time Human Detection and Distance Estimation
- Captures synchronized RGB and depth streams from Intel RealSense D435.
- Runs YOLOv8 inference on RGB frames to detect human targets (`person` class).
- Applies lightweight frame-to-frame association (IoU-based tracking) to reduce bounding-box flicker.
- Estimates per-person distance by sampling aligned depth values inside each detected human ROI.
- Displays both RGB detection overlays and a color-mapped depth visualization.

### Multi-Profile Face Enrollment and Recognition (Web Workflow)
- Provides a browser-based interface for operator control:
  - Enroll up to 10 local face profiles
  - Delete a selected face profile
  - Reject duplicate face enrollment
  - Start recognition mode
  - Stop processing
- Performs guided multi-sample enrollment for each identity, then builds a stable template vector.
- Runs live recognition and labels detected faces as:
  - `人物1` ... `人物10`
  - `Unknown`
- Shows quality and confidence indicators to help users collect valid enrollment samples.

### Age-Group Estimation with Transfer Learning
- Adds a local age-group classifier based on EfficientNet-B0.
- Uses transfer learning from ImageNet and fine-tunes on public age datasets such as UTKFace and FairFace.
- Predicts five stable life-stage age groups instead of exact age:
  - Child: 0-12
  - Teen: 13-19
  - Young adult: 20-35
  - Middle age: 36-55
  - Senior: 56+
- Integrates age-group output with multi-face recognition, so the system can report identity, age group, similarity, and depth.
- Records the enrolled user's actual age locally and compares the predicted age group with the actual age group during recognition.
- Applies a one-level upward age-group calibration for the target deployment population and stabilizes predictions with multi-frame majority voting.

### Configurable Experimental Platform
- Centralizes all critical parameters in `config.py`:
  - Camera settings (resolution/FPS)
  - YOLO inference settings
  - Tracking thresholds
  - Depth filtering limits
  - Face quality and recognition thresholds
- Enables fast parameter sweeps for reproducible experiments.
- Supports controlled tuning for different lighting, distance, and hardware performance conditions.

### Local Persistence for Identity Profiles
- Stores enrolled identity embeddings to a local `.npz` profile file.
- Loads existing profiles at startup for repeated testing without re-enrollment.
- Includes profile version metadata to avoid accidental mismatch with legacy feature formats.

---

## System Architecture and Implementation Principles

### 1) Sensor Pipeline and Stream Alignment

The system initializes a RealSense pipeline with two streams:
- Color stream (BGR)
- Depth stream (Z16)

Depth is aligned to the color coordinate system (`rs.align(rs.stream.color)`), so every detection box in RGB can be mapped directly to the corresponding depth ROI. This alignment is essential for stable distance estimation; without it, RGB/depth pixels refer to different spatial rays and produce incorrect measurements.

### 2) Human Detection Subsystem

`test.py` implements a real-time detection loop:
1. Pull aligned frames.
2. Run YOLOv8 on RGB frame.
3. Keep human detections above confidence/size thresholds.
4. Associate detections with existing tracks using IoU.
5. Update track states (`hits`, `misses`, confidence, distance).
6. Render stable overlays and summary counters.

Design choice:
- A lightweight IoU tracker is used instead of a heavy MOT stack to keep dependencies simple and latency low for a graduation-project prototype.

### 3) Depth Estimation Strategy (ROI Median Filtering)

Instead of using one center pixel (`get_distance(cx, cy)`), the project uses an ROI median method:
- Convert raw depth to meters using depth scale.
- Extract depth values inside the detection ROI.
- Remove invalid values (too near, too far, zero/holes).
- Compute median of remaining valid values.

Why median:
- Robust to outliers at object edges.
- Less sensitive to sensor noise, specular surfaces, and hole artifacts.
- Produces more stable temporal readings for on-screen labels and downstream logic.

### 4) Face Feature Pipeline

`face_identity.py` uses `facenet-pytorch`:
- **MTCNN** for face detection/alignment.
- **InceptionResnetV1** for 512-D embedding extraction.

Workflow:
1. Detect faces.
2. Select the dominant face candidate.
3. Align/crop and infer embedding.
4. Normalize embedding with L2 norm.
5. Return embedding + detection probability + quality metrics.

Rationale:
- This architecture provides stronger identity discrimination than handcrafted descriptors.
- L2-normalized embeddings allow cosine similarity to be used directly and efficiently.

### 5) Enrollment Logic (Template Construction)

During enrollment mode:
- Frames are sampled with a time interval to avoid highly redundant near-identical samples.
- A quality gate validates each sample (minimum detection confidence, optional blur threshold).
- Valid embeddings are appended to an identity buffer.
- Once the target sample count is reached:
  - Average embedding is computed.
  - The mean vector is re-normalized.
  - Template is saved to local profile storage.

Benefits:
- Multi-sample averaging reduces one-frame noise.
- Quality gating prevents low-information samples from contaminating templates.

### 6) Recognition Logic (Closed-set with Unknown Rejection)

In recognition mode:
1. Extract current face embedding.
2. Compute cosine similarity with each enrolled template, up to 10 local profiles.
3. Select highest score as candidate identity.
4. Compare against recognition threshold:
   - Above threshold: assign the best matched identity
   - Below threshold: output `Unknown`

This is a practical closed-set recognition design with open-set rejection behavior (via threshold).

Before saving a newly enrolled template, the system also compares it with existing templates. If the best similarity exceeds the duplicate threshold, enrollment is rejected and the original profile set is preserved.

### 7) Age-Group Estimation Logic

The age module uses `age_estimator.py` and `train_age_model.py`:

1. The detected face ROI is cropped from the RGB frame.
2. The crop is resized to 224x224 and normalized with ImageNet statistics.
3. EfficientNet-B0 outputs probabilities over five life-stage age groups.
4. The best age group and confidence are attached to the recognition result.
5. A post-processing calibration shifts the predicted age group upward by one class to reduce underestimation in the target deployment population.
6. A sliding-window majority vote over recent frames stabilizes the displayed age group.
7. If the recognized identity has an enrolled actual age, the system checks whether the calibrated and smoothed age group matches the actual age group and reports correct/incorrect.

The model is trained locally by fine-tuning ImageNet-pretrained EfficientNet-B0 on UTKFace and/or FairFace. The training pipeline uses a two-stage strategy: first freezing the backbone and training the classifier head, then unfreezing the network for full fine-tuning. It also reports exact accuracy, adjacent-group accuracy, and macro recall.
In the current project workspace, the default UTKFace training directory is configured as `UTKFace`, so the training script can be launched without passing a dataset path.

### 8) Web Interaction and API Layer

`face_app.py` + `templates/index.html` implement a simple control plane:
- Video stream endpoint uses MJPEG multipart response.
- Control endpoints:
  - Start enrollment (`/api/enroll/start`)
  - Delete profile (`/api/profile/delete`)
  - Start recognition (`/api/recognize/start`)
  - Stop (`/api/stop`)
- Status endpoint (`/api/status`) exposes runtime mode, enrollment progress, profile readiness, and the current number of enrolled profiles.

Frontend behavior:
- Sends control actions via `fetch`.
- Polls status periodically to update UI badges and mode labels.
- Keeps operator workflow explicit and repeatable for demo/testing.

### 9) State Management and Runtime Modes

The application uses shared runtime state with locking for thread safety:
- `idle`: camera stream active, no enrollment/recognition decision output.
- `enroll`: sample collection and template update for selected person.
- `recognize`: live identity classification.

State variables include:
- Current mode
- Target enrollment identity
- Last enrollment timestamp
- Progress buffers per identity
- Current status text for UI feedback

### 10) Parameterization and Reproducibility

Key reproducibility controls are externalized in `config.py`:
- Detection thresholds
- Tracking persistence values
- Depth validity range
- Face quality gates
- Recognition threshold
- Enrollment sample count and interval

This allows structured experiment logs such as:
- “Threshold = 0.42 under indoor white light”
- “Min blur variance = 40 improved false-accept behavior”

### 11) Engineering Trade-offs

Current design favors:
- Fast setup
- Explainable algorithmic flow
- Practical robustness under classroom/lab conditions

Known trade-offs:
- Lightweight IoU tracking is simpler than full MOT and may fail under heavy occlusion.
- Two-identity closed-set design is ideal for controlled demo scenarios, but not for large-scale identity management.
- Recognition quality still depends on enrollment posture, illumination, and camera placement.

---

## File-Level Implementation Map

### `config.py`
- Global constants for camera, YOLO, tracking, depth filtering, and face-recognition thresholds.

### `realsense_utils.py`
- RealSense depth-scale helper.
- ROI median depth function with validity masking.

### `test.py`
- Real-time human detection + tracking + depth overlay demo in OpenCV windows.

### `face_identity.py`
- Face embedding backend using MTCNN + InceptionResnetV1.
- Embedding normalization and quality-related outputs.

### `age_estimator.py`
- Loads the trained EfficientNet-B0 age-group classifier.
- Returns age-group label, age range, and confidence for each face crop.

### `train_age_model.py`
- Trains the age-group classifier from UTKFace and/or FairFace.
- Saves `age_model_effnet_b0.pth` for direct use by the web application.

### `face_app.py`
- Flask server, mode switching, enrollment/recognition orchestration, persistence, MJPEG streaming.

### `templates/index.html`
- Browser UI for control actions and status display.

