import numpy as np
from fastapi import FastAPI, File, UploadFile
import fastapi.responses
import os
import cv2
import torch

fastapi.responses.JSONResponse

bind_ip_address = "127.0.0.1"
bind_port = 8001
detection_threshold = 0.3

# RestAPI request example for local file example.jpg
# curl -X "POST"  "http://127.0.0.1:8001/classify-image/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@example.jpg;type=image/jpeg"

model = torch.hub.load(
    "yolov5", 'custom', path="real_world_pytorch_model.pt", source='local')
model.conf = detection_threshold  # Threshold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()


@app.post("/classify-image/")
async def upload_file(file: UploadFile = File(...)):
    try:
        img = await file.read()
        np_buffer = np.frombuffer(img, np.uint8)
        image = cv2.imdecode(np_buffer,
                             cv2.IMREAD_COLOR)[..., ::-1]
    except:
        return fastapi.responses.JSONResponse(content={"filename": file.filename, "message": "Cannot decode image"})
    finally:
        pass

    result = model(image)
    return fastapi.responses.JSONResponse(content={"filename": file.filename, "message": "File uploaded successfully", "shape": image.shape, "xywh": result.pandas().xywh[0].to_json(), "xyxy": result.pandas().xyxy[0].to_json()})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=bind_ip_address, port=bind_port)
