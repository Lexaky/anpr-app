import os
import io
import cv2
import numpy as np
import base64
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms

app = FastAPI()
templates = Jinja2Templates(directory="templates")

PLATE_MODEL_PATH = "models/plate.pt"
CHAR_MODEL_PATH = "models/char.pt"

if not os.path.exists(PLATE_MODEL_PATH) or not os.path.exists(CHAR_MODEL_PATH):
    raise FileNotFoundError("Файлы моделей plate.pt и char.pt должны находиться в директории models/")

class CRNN(nn.Module):
    def __init__(self, n_classes, hidden=256, lstm_layers=2, img_h=32, img_w=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
        )
        with torch.no_grad():
            feat = self.cnn(torch.zeros(1, 1, img_h, img_w))
            lstm_in = feat.shape[1] * feat.shape[2]
        self.lstm = nn.LSTM(lstm_in, hidden, lstm_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        return self.fc(self.lstm(x.permute(0, 3, 1, 2).reshape(b, w, c * h))[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    plate_model = YOLO(PLATE_MODEL_PATH)
except Exception:
    plate_model = YOLO("yolov8n.pt")
    ckpt_plate = torch.load(PLATE_MODEL_PATH, map_location="cpu")
    state_dict_plate = ckpt_plate.get("model", ckpt_plate)
    plate_model.model.yaml['nc'] = 1
    plate_model.names = {0: 'plate'}
    plate_model.model.load_state_dict(state_dict_plate, strict=False)

def load_crnn(path):
    ckpt = torch.load(path, map_location=device)
    m = CRNN(len(ckpt["charset"]) + 1, ckpt.get("hidden", 256), ckpt.get("lstm_layers", 2),
             ckpt.get("img_h", 32), ckpt.get("img_w", 128)).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, ckpt["charset"], ckpt["img_h"], ckpt["img_w"]

char_model, crnn_charset, crnn_h, crnn_w = load_crnn(CHAR_MODEL_PATH)

def decode_crnn(out, charset):
    idx = {i + 1: c for i, c in enumerate(charset)}
    prev, chars = -1, []
    for p in out.argmax(2).squeeze(0).tolist():
        if p and p != prev:
            chars.append(idx.get(p, ""))
        prev = p
    return "".join(chars)

def predict_crnn(model, crop_bgr, charset, img_h, img_w):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    tf = transforms.Compose([
        transforms.Resize((img_h, img_w)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    x = tf(Image.fromarray(gray)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        return decode_crnn(model(x), charset)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_img = img.copy()

    plate_results = plate_model(img, verbose=False)[0]
    
    plate_found = False
    recognized_text = "Номерной знак не обнаружен"
    
    if len(plate_results.boxes) > 0:
        plate_found = True
        best_plate = max(plate_results.boxes, key=lambda b: b.conf[0].item())
        x1, y1, x2, y2 = map(int, best_plate.xyxy[0].tolist())
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, "PLATE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        h, w, _ = orig_img.shape
        pad = 5
        crop_y1, crop_y2 = max(0, y1 - pad), min(h, y2 + pad)
        crop_x1, crop_x2 = max(0, x1 - pad), min(w, x2 + pad)
        crop_img = orig_img[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop_img.size > 0:
            try:
                recognized_text = predict_crnn(char_model, crop_img, crnn_charset, crnn_h, crnn_w)
            except Exception:
                recognized_text = "Ошибка работы CRNN"

            if not recognized_text:
                recognized_text = "Символы не распознаны"

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "plate_found": plate_found,
        "image": f"data:image/jpeg;base64,{img_base64}",
        "text": recognized_text
    }