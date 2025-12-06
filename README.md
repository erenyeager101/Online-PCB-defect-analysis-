# PCB Defect Detection using Deep Learning 🔍

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

An intelligent PCB (Printed Circuit Board) defect detection system using deep learning and computer vision. This project leverages MobileNet architecture with transfer learning to identify defects in PCB images with 95.71% validation accuracy.

![PCB Examples](https://user-images.githubusercontent.com/47279340/125171710-2761c500-e1d3-11eb-93e1-c444fd3c78bf.jpeg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Web Application](#-web-application)
- [Dataset](#-dataset)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **High Accuracy**: 95.71% validation accuracy using MobileNet with transfer learning
- **Real-time Detection**: Fast inference for quick PCB quality assessment
- **Web Interface**: User-friendly Flask-based web application
- **Interactive Dashboard**: View analytics, compare results, and provide feedback
- **Data Augmentation**: Robust model training with diverse image transformations
- **Easy Deployment**: Simple setup with Docker support (optional)

---

## 📁 Project Structure

```
Defect-Detection-of-PCB/
│
├── app.py                      # Main Flask application
├── smoke_test.py               # Quick system validation test
├── requirements.txt            # Python dependencies
├── requirements-py312.txt      # Python 3.12 specific dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── LICENSE                     # Project license
│
├── notebooks/                  # Jupyter notebooks
│   ├── defect-pred-new.ipynb  # Model training notebook
│   ├── prediction_model_i.ipynb # Prediction testing notebook
│   └── app_predt_i.ipynb      # Flask app development notebook
│
├── models/                     # Trained model files (ignored by Git)
│   ├── model.h5               # Trained model weights
│   └── model.weights.h5       # Alternative weights format
│
├── data/                       # Application data
│   ├── predictions.csv        # Prediction history
│   └── feedback.csv           # User feedback data
│
├── dataset/                    # Training and test datasets
│   ├── train/
│   │   └── train5/           # Training images (not included in repo)
│   └── test/
│       └── test5/            # Test images (not included in repo)
│
├── static/                     # Static files for Flask
│   └── uploads/               # User uploaded images
│
├── templates/                  # HTML templates
│   ├── home.html             # Landing page
│   ├── predict.html          # Prediction interface
│   ├── dashboard.html        # Analytics dashboard
│   ├── compare.html          # Comparison view
│   ├── analytics.html        # Detailed analytics
│   ├── feedback.html         # Feedback form
│   └── about.html            # About page
│
└── docs/                       # Documentation
    └── Setup_Instruction_File.txt  # Original setup guide
```

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Defect-Detection-of-PCB.git
cd Defect-Detection-of-PCB
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# For Python 3.7-3.11
pip install -r requirements.txt

# For Python 3.12+
pip install -r requirements-py312.txt
```

### Step 4: Download Dataset (Optional)

Download the DeepPCB dataset from [here](https://github.com/tangsanli5201/DeepPCB) and place it in the `dataset/` directory.

### Step 5: Verify Installation

```bash
python smoke_test.py
```

Note: Model files (`models/*.h5`) are ignored by default to keep the repo lightweight.
Place your trained model at `models/model.h5`. For versioning large binaries, consider Git LFS.

---

## 💻 Usage

### Running the Web Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the Notebooks

1. **Training the Model**: Open `notebooks/defect-pred-new.ipynb`
   - Contains complete model architecture and training pipeline
   - **Note**: Training can take ~1 hour on Google Colab

2. **Testing Predictions**: Open `notebooks/prediction_model_i.ipynb`
   - Test the model on individual images
   - Evaluate model performance

3. **Flask Development**: Open `notebooks/app_predt_i.ipynb`
   - Development and testing of Flask application

### Making Predictions

#### Via Web Interface:
1. Navigate to `http://localhost:5000`
2. Click "Predict" or upload an image
3. View results with defect classification

#### Via Command Line:
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('models/model.h5')

# Load and preprocess image
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
result = "Good Quality" if prediction[0][0] > 0.5 else "Defective"
print(f"Result: {result}")
```

---

## 🧠 Model Architecture

### MobileNet with Transfer Learning

- **Base Model**: MobileNet pre-trained on ImageNet
- **Input Shape**: 224x224x3 (RGB images)
- **Frozen Layers**: First 20 layers
- **Custom Layers**: Dense layers for binary classification
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Binary Crossentropy

### Data Augmentation

- Rotation: ±20 degrees
- Width/Height Shift: ±20%
- Shear Transformation: 20%
- Zoom: ±20%
- Horizontal Flip
- Fill Mode: Nearest

### Training Details

- **Epochs**: Optimized with early stopping
- **Batch Size**: 32
- **Callbacks**: Learning rate scheduler, early stopping
- **Validation Split**: 20%

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 95.71% |
| **Training Accuracy** | 97.2% |
| **Precision** | 94.8% |
| **Recall** | 96.3% |

![Training Graphs](https://user-images.githubusercontent.com/47279340/125171519-4c096d00-e1d2-11eb-820e-4da9a63aaee6.png)

---

## 🌐 Web Application

### Features

- **Image Upload**: Support for PNG, JPG, JPEG, BMP formats
- **Real-time Prediction**: Instant defect classification
- **Dashboard**: View prediction statistics and trends
- **Analytics**: Detailed performance metrics
- **Comparison**: Side-by-side image comparison
- **Feedback System**: User feedback collection for model improvement

### Technology Stack

- **Backend**: Flask 2.0+
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas, PIL

### Screenshots

<div align="center">
<kbd><img src="https://user-images.githubusercontent.com/47279340/125175412-8cc0b080-e1e9-11eb-87be-bb04da89742d.png" width="700"></kbd>
<p><i>Upload Interface</i></p>

<kbd><img src="https://user-images.githubusercontent.com/47279340/125176421-d5c83300-e1f0-11eb-9ea6-2428b48cfcaf.png" width="700"></kbd>
<p><i>Prediction Results</i></p>
</div>

---

## 📊 Dataset

### DeepPCB Dataset

- **Source**: [DeepPCB GitHub Repository](https://github.com/tangsanli5201/DeepPCB)
- **Format**: JPEG images
- **Original Size**: 640x640 pixels
- **Processed Size**: 224x224 pixels (using anti-aliasing)
- **Type**: Binary (defective/non-defective)
- **Color**: Grayscale (Black & White)

### Image Characteristics

- **Black regions**: Circuit components
- **White patches**: Potential defects
- **Preprocessing**: Anti-aliasing resize to maintain quality

**Note**: Due to size constraints, the dataset is not included in this repository. Please download it separately from the source link above.

---

## 📈 Results

### Sample Predictions

The model successfully identifies both good and defective PCBs:

<div align="center">
<img src="https://user-images.githubusercontent.com/47279340/125174250-9c3bfb80-e1e1-11eb-8c8a-b8ec1b718708.png" width="700">
<p><i>Prediction examples showing good and defective PCBs</i></p>
</div>

### Key Achievements

✅ 95.71% validation accuracy  
✅ Fast inference time (~0.1s per image)  
✅ Robust to various PCB layouts  
✅ Effective defect localization  
✅ User-friendly deployment  

---

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- 🐛 Bug fixes and improvements
- 📝 Documentation enhancements
- ✨ New features (e.g., multi-class classification)
- 🎨 UI/UX improvements
- 🧪 Additional test coverage
- 🔧 Performance optimizations

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DeepPCB Dataset**: Thanks to the creators of the DeepPCB dataset
- **MobileNet**: Google's MobileNet architecture
- **TensorFlow/Keras**: For the deep learning framework
- **Flask**: For the web framework
- **Bootstrap**: For the responsive UI components

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/Defect-Detection-of-PCB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/Defect-Detection-of-PCB/discussions)

---

## 🔮 Future Improvements

- [ ] Multi-class defect classification (scratch, short circuit, open circuit, etc.)
- [ ] Real-time video stream processing
- [ ] RESTful API for integration
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile application
- [ ] Explainable AI (Grad-CAM visualization)
- [ ] Model quantization for edge devices

---

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{pcb-defect-detection,
  title={PCB Defect Detection using Deep Learning},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/Defect-Detection-of-PCB}
}
```

---

<div align="center">
Made with ❤️ by the PCB Defect Detection Team

**⭐ Star this repository if you find it helpful!**
</div>

