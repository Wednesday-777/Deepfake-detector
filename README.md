# Deepfake Detector
A deepfake image classifier with built-in interpretability - upload a face image and see not just whether it's real or AI-generated, via a Grad-CAM heatmap showing which regions drove the decision.
# Live demo: https://deepfake-detector-plum.vercel.app 
# API: https://deepfake-detector-hn8h.onrender.com
# How it works
A face image is uploaded through the React frontend

FastAPI backend runs it through a fine-tuned EfficientNetB0 classifier

Grad-CAM computes which pixels most influenced the prediction

The result - label, confidence, and heatmap overlay - is returned and displayed
# Model
Architecture: EfficientNetB0 (ImageNet-pretrained), fine-tuned in two phases

Dataset: 140k Real and Fake Faces (StyleGAN-generated fakes) -> (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

Test performance: 95.14% accuracy · 0.9926 AUC

Interpretability: Grad-CAM on the final convolutional layer
# Tech stack
Model - TensorFlow / Keras, EfficientNetB0, Grad-CAM

Backend -	FastAPI

Frontend - React (Vite)

Backend hosting - Render (free tier)

Frontend hosting - Vercel
# Limitations
Trained only on StyleGAN-generated fakes - does not reliably detect diffusion-generated images (e.g., Gemini, Midjourney, Stable Diffusion).Confirmed via testing: a Gemini-generated passport-style photo was classified as real with 89.4% confidence despite ideal framing.

Built for cropped, front-facing face images - untested on full-body or non-face content.

Recall on fakes (98%) is higher than on real images (93%) - the model leans toward flagging uncertain cases as fake rather than missing one.

Free-tier backend hosting may take up to ~60s to respond on first request after inactivity (cold start).
