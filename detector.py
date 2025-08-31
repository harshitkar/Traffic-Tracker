import torch
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import LoadStreams, LoadImages
from config import *
from utils import select_device

def load_model():
    device = select_device(DEVICE)
    model = DetectMultiBackend(YOLO_MODEL, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = IMG_SIZE
    return model, stride, names, device, imgsz

def get_dataloader(source, imgsz, stride):
    if source.isnumeric() or source.endswith(".txt") or source.startswith("rtsp") or source.startswith("http"):
        return LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
    else:
        return LoadImages(source, img_size=imgsz, stride=stride, auto=True)

def run_inference(model, im, augment=False, visualize=False):
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, MAX_DET)
    return pred
