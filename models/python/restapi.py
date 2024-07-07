import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
import fastapi.responses
import os
import cv2
import torch
import argparse
import re
import logging
import uvicorn

def log(message):
    print(message)

def validate_ip_address(ip_address):
    ip_pattern = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    if ip_pattern.match(ip_address):
        return True
    return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script with IP address, port, and detection threshold parameters.")
    parser.add_argument('--bind_ip_address', type=str, default="127.0.0.1", help='The IP address to bind to')
    parser.add_argument('--bind_port', type=int, default=8001, help='The port to bind to')
    parser.add_argument('--detection_threshold', type=float, default=0.3, help='The detection threshold')

    args = parser.parse_args()

    try:
        assert validate_ip_address(args.bind_ip_address), "Invalid IP address format"
    except AssertionError as error:
        log(f"Error: {error}")
        log("Using default IP address: 127.0.0.1")
        args.bind_ip_address = "127.0.0.1"

    try:
        assert 0 <= args.detection_threshold <= 1, "Detection threshold must be between 0 and 1"
    except AssertionError as error:
        log(f"Error: {error}")
        log("Using default detection threshold: 0.3")
        args.detection_threshold = 0.3

    return args

fastapi.responses.JSONResponse

model = torch.hub.load(
    "yolov5", 'custom', path="real_world_pytorch_model.pt", source='local')

app = FastAPI()


@app.post("/classify-image/")
async def upload_file(file: UploadFile = File(...)):
    try:
        img = await file.read()
        np_buffer = np.frombuffer(img, np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)[..., ::-1]
    except:
        return fastapi.responses.JSONResponse(content={
            "filename": file.filename,
            "message": "Cannot decode image"
        })
    finally:
        pass

    result = model(image)
    return fastapi.responses.JSONResponse(content={
        "filename": file.filename, 
        "message": "File uploaded successfully", 
        "shape": image.shape, 
        "xywh": result.pandas().xywh[0].to_json(), 
        "xyxy": result.pandas().xyxy[0].to_json()
    })


if __name__ == "__main__":
    args = parse_arguments()
    print(f"""
curl -X "POST"  "http://{args.bind_ip_address}:{args.port}/classify-image/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@example.jpg;type=image/jpeg"
""")
    model.conf = args.detection_threshold  # Threshold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uvicorn.run(app, host=args.bind_ip_address, port=args.bind_port)
