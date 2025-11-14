# YOLOv12n Thermal Video Detector

This repository contains a **YOLOv12n** model fine-tuned for **object detection in railway thermal videos**. The model is optimized to recognize key railway infrastructure elements under thermal conditions and is designed to run efficiently even on lightweight GPUs such as Google Colab.

---

## 🚀 Try It on Google Colab (with GPU)

You can test the model in real time using the interactive Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Simone2408/yolov12n-thermal-video-detector/blob/main/notebooks/yolov12n_thermal_demo.ipynb)

Inside the notebook you can:

* Automatically load the model
* Upload your own thermal video
* Run GPU-accelerated inference
* Visualize the output

---

## 🧠 Detected Classes

The YOLOv12n model is trained to recognize the following railway-related thermal classes:

* **train**
* **signal**
* **railway switch**
* **pole**
* **balise**
* **overhead support arm**

---

## 📁 Dataset

The model was trained on a non-public thermal railway dataset containing annotated samples for the six classes listed above.

The dataset was split using a **70/20/10 ratio**:

| Split          | Percentage | Purpose                                |
| -------------- | ---------- | -------------------------------------- |
| **Train**      | 70%        | Model training                         |
| **Validation** | 20%        | Hyperparameter tuning  |
| **Test**       | 10%        | Final evaluation & official metrics    |

Annotations follow the **YOLO format** with normalized bounding boxes.

---

## 📂 Project Structure

```
yolov12n-thermal-video-detector/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── weights/
│   └── best.pt
│
├── src/
│   ├── video_inference.py
│   ├── utils.py
│   └── __init__.py
│
├── notebooks/
│   └── yolov12n_thermal_demo.ipynb
│
├── images/
│   ├── loss.png
│   ├── example_thermal1.jpg
│   └── example_thermal2.jpg
│
├── outputs/
│   └── .gitkeep
```

---

## 🧩 Local Installation

> ⚠️ Real-time inference on CPU may be slow. GPU execution is recommended.

### 1. Create a Virtual Environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Clone the Repository and Install Dependencies

```bash
git clone https://github.com/Simone2408/yolov12n-thermal-video-detector.git
cd yolov12n-thermal-video-detector
pip install -r requirements.txt
```

### 3. Download the Weights

Download the `best.pt` file:

[![Download Weights](https://img.shields.io/badge/Download-best.pt-blue?style=for-the-badge)](https://drive.google.com/uc?export=download&id=1V6x8ROG5AGGCQ5PUdtZkr4UnbfXqBrZb)

---

## 📊 Model Performance

Metrics computed on the **test set** at epoch 80:

| Metric           | Value     |
| ---------------- | --------- |
| **mAP@0.5**      | **0.905** |
| **mAP@0.5:0.95** | **0.552** |
| **Precision**    | **0.898** |
| **Recall**       | **0.840** |

---

## 📉 Training Progress

![Loss Curve](images/loss.png)

This plot shows the evolution of both training and validation loss.

---

## 📈 Evaluation Curves

### Recall–Confidence

![R Curve](images/R_curve.png)

### Precision–Recall

![PR Curve](images/PR_curve.png)

### Precision–Confidence

![P Curve](images/P_curve.png)

### F1–Confidence

![F1 Curve](images/F1_curve.png)

---

## 🧩 Confusion Matrices

### Absolute

![Confusion Matrix](images/confusion_matrix.png)

### Normalized

![Confusion Matrix Normalized](images/confusion_matrix_normalized.png)

---

## 🔥 Thermal Inference Examples

![Example 1](images/example_thermal1.jpg)
![Example 2](images/example_thermal2.jpg)

---

## ⚠️ Limitations

* Very small objects may be difficult to detect
* A GPU is recommended for high frame rates

---

## 📄 Citations

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@software{yolov12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLOv12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {https://github.com/sunsmarterjie/yolov12},
  license = {AGPL-3.0}
}
```

---

## 🤝 Contributing

Contributions and pull requests are welcome! If you use this model in real applications, feel free to reach out — it's always exciting to see real-world implementations.


## 📄 License

MIT License
