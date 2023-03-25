import pprint
import torch
import argparse

def green_msg (msg):
    print('\x1b[6;30;42m' + msg + '\x1b[0m')

def yellow_msg (msg):
    print('\x1b[6;30;43m' + msg + '\x1b[0m')


parser = argparse.ArgumentParser(description='UFO-AI-Example-Script')
parser.add_argument('--img', nargs='+', type=str, help='Image file', required=True)
parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for detection')

args = parser.parse_args()

model = torch.hub.load("yolov5", 'custom', path="pytorch_model.pt", source='local')
model.conf = args.threshold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in args.img:
    result = model(i)
    green_msg(i + ":")
    yellow_msg("xywh as CSV:")
    print(result.pandas().xywh[0])

    json = result.pandas().xyxy[0].to_json()
    yellow_msg("xyxy as JSON:")
    print(json)
