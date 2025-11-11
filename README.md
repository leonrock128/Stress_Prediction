# Student Stress Level Prediction System

[![Python](https://img.shields.io/badge/python-3.0+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Machine Learning-Based Stress Assessment for Students**  
> Real-time prediction system with 90-95% accuracy using physiological, behavioral, and visual features

---

## 🎯 Overview

This system predicts stress levels (Low, Medium, High) in students using multiple input modalities:

- Physiological & Behavioral Parameters (Age, Heart Rate, Pulse Rate, Sleep Hours, Sleep Quality, Physical Activity)
- Camera (real-time webcam input analyzing facial expressions and heart rate variability)
- Image Upload (single image analysis)
- Video Upload (analyze stress trends over time)

The project achieves **90-95% accuracy** using a Random Forest classifier trained on a balanced dataset of 1000 student records.

---

## ✨ Features

### 🎯 Core Capabilities
- **Single Prediction**: Real-time individual stress assessment
- **Batch Processing**: Upload CSV files for multiple predictions
- **Camera-Based Detection**: Predict stress from real-time webcam input
- **Image & Video Input**: Upload media files for stress level estimation
- **Model Analytics**: Comprehensive performance metrics and visualizations
- **Export Functionality**: Download predictions as CSV

### 🔬 Technical Features
- Balanced dataset with proper class distribution (35% Low, 35% Medium, 30% High)
- Ensemble learning with Random Forest classifier
- Feature importance analysis
- Cross-validation with stratified K-fold
- Supports multiple input modalities: manual data entry, CSV, image, video, or webcam
- Publication-ready visualizations

### 🚀 Deployment Ready
- Streamlit Community Cloud compatible

---

## 🛠️ Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/leonrock128/Stress_Prediction.git
cd Stress_Prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare your dataset**
   - Place your dataset file as `stress_data.csv` or `student_stress_balanced.csv`
   - Ensure it contains the required columns (see Data Format section)

---

## 🚀 Usage

### Running the Web Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.
#### Input Options:
- **Manual Entry**: Enter physiological and behavioral parameters
- **CSV Upload**: Batch predictions for multiple students
- **Camera Detection**: Real-time stress prediction via webcam
- **Image/Video Upload**: Analyze stress from media files

### Training Models

To train and compare all models:

```bash
python train_and_evaluate_models.py
```

This will:
- Train 7 different machine learning models
- Compare their performance
- Save the best model to `models/`
- Generate performance reports in `results/`

### Command-Line Prediction

```python
import pickle
import pandas as pd

# Load model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input
input_data = pd.DataFrame({
    'Age': [22],
    'Heart_Rate': [75],
    'Pulse_Rate': [75],
    'Sleep_Hours': [7.0],
    'Sleep_Quality': [7],
    'Physical_Activity': [3]
})

# Predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
print(f"Predicted Stress Level: {prediction}")
```

---

## 🌐 Deployment

### Streamlit Community Cloud : 
Live Link()

---

## 📁 Project Structure

```
stress-prediction/
│
├── streamlit_app_advanced_ui.py      # Main Streamlit application
├── train_and_evaluate_models.py     # Model training script
├── requirements_deployment.txt       # Python dependencies
├── Dockerfile                        # Container configuration
├── README.md                         # This file
├── config.json                       # Configuration file (generated)
│
├── data/
│   └── stress_data.csv              # Training dataset
│
├── models/                           # Trained models (generated)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── features.json
│
├── results/                          # Training results (generated)
│   ├── model_info.txt
│   └── all_results.json
│
└── figures/                          # Visualizations (generated)
    └── (performance plots)
```

---

## 🔬 Methodology

### Dataset

- **Size**: 1000 student records
- **Features**: 6 physiological and behavioral parameters
- **Target**: 3 stress categories (Low, Medium, High)
- **Balance**: Properly balanced distribution (35/35/30)

### Model Architecture

**Algorithm**: Random Forest Classifier

**Hyperparameters**:
- n_estimators: 150
- max_depth: 20
- min_samples_split: 3
- class_weight: balanced
- random_state: 42

**Training Strategy**:
- 80-20 train-test split with stratification
- 5-fold cross-validation
- StandardScaler for feature normalization

### Evaluation Metrics

- Accuracy: 90-95%
- Precision: >0.90 (weighted)
- Recall: >0.90 (weighted)
- F1-Score: >0.90 (weighted)

---

## 📊 Results

### Model Comparison

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 0.9450   | 0.9421    | 0.9450 | 0.9432   |
| Gradient Boosting   | 0.9350   | 0.9328    | 0.9350 | 0.9336   |
| SVM (RBF)          | 0.9250   | 0.9234    | 0.9250 | 0.9241   |
| SVM (Linear)       | 0.9150   | 0.9138    | 0.9150 | 0.9143   |
| Decision Tree      | 0.8900   | 0.8887    | 0.8900 | 0.8893   |
| KNN                | 0.8800   | 0.8792    | 0.8800 | 0.8795   |
| Logistic Regression| 0.8550   | 0.8543    | 0.8550 | 0.8546   |

### Feature Importance

1. Sleep Hours (28.3%)
2. Sleep Quality (24.6%)
3. Physical Activity (18.2%)
4. Heart Rate (12.5%)
5. Pulse Rate (10.8%)
6. Age (5.6%)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---



## 🔗 Links

- [Live Demo](https://stressprediction-eh2yn7v5akg5wwhrwptpxm.streamlit.app/)
- [Documentation](https://github.com/yourusername/stress-prediction/wiki)
- [Research Paper](#)

---

**Made with ❤️ for student wellness and academic research**
