# Deepfake Detection System

A hybrid deep learning system that detects deepfake images using EfficientNet-B4 and Vision Transformer (ViT) with Grad-CAM explainability visualizations.

## Motivation

Deepfakes have emerged as a growing threat to society — enabling misinformation, identity fraud, and manipulation at scale. This project is my attempt to build a robust, explainable detection system using a hybrid CNN + Transformer architecture that reflects the kind of applied AI being built in industry today.

## Architecture

```
Input Image → [EfficientNet-B4] ─┐
                                  ├→ Feature Fusion → Classifier → Real/Fake
Input Image → [ViT-Base/16]    ───┘                        ↓
                                                      Grad-CAM Heatmap
```

## Features

- Hybrid CNN + Transformer architecture for robust detection
- Grad-CAM heatmaps showing where the model detects manipulation
- FastAPI REST backend with auto-generated Swagger docs
- React frontend with sparkle effect UI
- Confidence scores with probability breakdown

## Tech Stack

**ML/DL:** PyTorch, EfficientNet-B4, ViT-Base/16, timm, OpenCV  
**Backend:** FastAPI, Uvicorn, Pillow, NumPy  
**Frontend:** React, Vite, Tailwind CSS  
**Training:** Google Colab (T4 GPU)

## Project Structure

```
deepfake-detector/
├── backend/
│   ├── model.py          # Hybrid EfficientNet-B4 + ViT architecture
│   ├── predict.py        # Inference pipeline
│   ├── gradcam.py        # Grad-CAM explainability
│   ├── main.py           # FastAPI server
│   └── requirements.txt
├── frontend/
│   └── deepfake-ui/      # React application
└── README.md
```

## Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend/deepfake-ui
npm install
npm start
```

API docs available at: `http://localhost:8000/docs`

## Model Details

| Component | Details |
|-----------|---------|
| CNN Backbone | EfficientNet-B4 (1792-dim features) |
| Transformer | ViT-Base/16 (768-dim features) |
| Fusion | Concatenation → 2560-dim |
| Classifier | Linear(2560→512→128→2) |
| Input Size | 224×224 RGB |
| Output | Real / Fake + confidence score |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API status |
| POST | /detect | Upload image → get prediction + heatmap |
| GET | /health | Health check |

## Roadmap

- [x] Hybrid model architecture
- [x] Grad-CAM integration
- [x] FastAPI backend
- [x] React frontend
- [ ] Model training on real dataset
- [ ] Deployment on Hugging Face Spaces
- [ ] Video deepfake support

## Author

**Harshit Shastri**  
Final Year AI/ML Student, College of Engineering Roorkee  
GitHub: [@harshitshastri2003](https://github.com/harshitshastri2003)
