import time
import io
import json
import os
import torch
import torchaudio
import torch.nn as nn
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch.nn.functional as F

# Архітектура моделі
class AudioCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 8, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

app = FastAPI(title="IoT Audio Control API")

CLASSES = ['yes', 'no', 'up', 'down']
MODEL_VERSION = "1.0.0"

# Шлях до моделі налаштовується через змінну середовища
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")

# Ініціалізація моделі при старті сервера
model = AudioCNN(num_classes=len(CLASSES))
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
model.eval()
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    accuracy = 0.0
    try:
        with open("artifacts/metrics.json", "r") as f:
            metrics = json.load(f)
            accuracy = metrics.get("accuracy", 0.0)
    except FileNotFoundError:
        pass

    model_size_kb = os.path.getsize(MODEL_PATH) / 1024 if os.path.exists(MODEL_PATH) else 0

    return {
        "architecture": "AudioCNN (2x Conv2d, 2x Linear) + Dynamic Quantization",
        "classes": CLASSES,
        "model_size_kb": round(model_size_kb, 2),
        "accuracy": accuracy,
        "model_version": MODEL_VERSION
    }

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    start_time = time.perf_counter()

    try:
        audio_bytes = await file.read()
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading audio file: {str(e)}")

    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    elif waveform.shape[1] > 16000:
        waveform = waveform[:, :16000]

    mel_spec = mel_transform(waveform).unsqueeze(0)

    with torch.no_grad():
        logits = model(mel_spec)
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        prediction_idx = torch.argmax(logits, dim=1).item()

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    probabilities = {CLASSES[i]: round(probs[i], 3) for i in range(len(CLASSES))}

    return {
        "label": CLASSES[prediction_idx],
        "confidence": round(probs[prediction_idx], 3),
        "probabilities": probabilities,
        "latency_ms": round(latency_ms, 2),
        "model_version": MODEL_VERSION
    }