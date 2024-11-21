from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8n.pt")
model.to(device=device)

results = model.train(data="data.yaml", epochs=20, imgsz=640)
 