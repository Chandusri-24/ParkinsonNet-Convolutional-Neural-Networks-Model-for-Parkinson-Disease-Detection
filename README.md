# Detection of Parkinson's Disease through Image and Speech Data
Parkinson’s disease (PD) is a progressive neurodegenerative disorder that affects motor skills, speech, and facial expressions. Early detection is crucial for timely intervention, but clinical diagnosis often occurs at later stages. This project uses machine learning and deep learning on both image and speech data to build a non-invasive, cost-effective, and accurate detection system.

## Features
- Image analysis using CNNs and transfer learning on facial micro-expressions, spiral drawings, and handwriting samples  
- Speech analysis with MFCCs, jitter, shimmer, pitch features using RNNs, LSTMs, and transformers  
- Multimodal fusion of image and speech models for improved accuracy  
- Deployment-ready pipeline using Streamlit or Flask  

## Methodology
1. Data Collection: UCI Parkinson’s speech dataset, spiral and handwriting datasets  
2. Preprocessing: Image resizing, normalization, augmentation; speech denoising, MFCC extraction, jitter/shimmer analysis, spectrogram conversion  
3. Modeling: CNNs (ResNet, EfficientNet, VGG) for images; RNN/LSTM/GRU/transformers for speech; late fusion for multimodal detection  
4. Evaluation: Accuracy, precision, recall, F1-score, ROC-AUC with cross-validation  

## Datasets
- [UCI Parkinson’s Speech Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)  
- [PhysioNet Spiral Drawings Dataset](https://physionet.org/content/spiralwaveform/1.0.0/)  

## Installation
```bash
git clone https://github.com/your-username/parkinsons-detection.git
cd parkinsons-detection
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt

## Usages
# Run image-based detection
python run_image_model.py --input data/spiral_sample.png

# Run speech-based detection
python run_speech_model.py --input data/sample_audio.wav

# Run multimodal detection
python run_fusion_model.py --image data/spiral_sample.png --audio data/sample_audio.wav

Results

Image-only models: 80–85% accuracy

Speech-only models: 82–88% accuracy

Multimodal fusion: 90%+ accuracy

Results

Image-only models: 80–85% accuracy

Speech-only models: 82–88% accuracy

Multimodal fusion: 90%+ accuracy

Future Work

Extend datasets for greater diversity and generalization

Real-time mobile application for continuous monitoring

Integration of handwriting dynamics using smart devices

References

UCI Machine Learning Repository – Parkinson’s Dataset

PhysioNet Spiral Drawings Database

Research papers on multimodal deep learning for Parkinson’s disease detection
