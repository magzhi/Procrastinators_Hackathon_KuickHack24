from ultralytics import YOLO
import torch
from torchvision.datasets.folder import ImageFolder
import pathlib



if __name__ == '__main__':
    model = YOLO(r"C:\Users\magzh\OneDrive\Desktop\HackathonDocs\FULL\runs\classify\train4\weights\best.pt")
    results=model.predict(r"C:\Users\magzh\OneDrive\Desktop\HackathonDocs\GeneratorApp\images\srb6.png")
    print(results[0].probs.top1,len(results),type(results))