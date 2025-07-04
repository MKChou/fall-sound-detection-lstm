﻿# Fall Sound Detection LSTM

This project utilizes an LSTM neural network for fall sound detection, suitable for smart healthcare and elderly safety applications.

## Directory Structure
```
fall-sound-detection-lstm/
│
├── src/
│   ├── main_fall_detection.py      # Main script
│   └── metrics_evaluator.py        # Model evaluation
├── export_test_data.py             # Export test data
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── .gitignore
├── training_results.png            # Training result plot
├── fall_detection_model.h5         # Keras model
├── fall_detection_model.tflite     # TFLite model
├── fall_detection_model.onnx       # ONNX model
```

## Installation
Install all dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
Place your wav files in the `data/` directory. The filename format should be:
- `*-01.wav` (fall)
- `*-02.wav` (non-fall)

## Training
```bash
python src/main_fall_detection.py
```

## Model Evaluation
```bash
python src/metrics_evaluator.py
```

## Output Description
- Training result plot: `training_results.png`
- Confusion matrix plot: `confusion_matrix.png`
- Trained model: `fall_detection_model.h5`
- Converted models: `fall_detection_model.tflite`, `fall_detection_model.onnx`
- ROC curve plot: `roc_curve.png`

## Notes
- Do not upload large datasets or model files directly to GitHub.
- To export test data, run `export_test_data.py`.


