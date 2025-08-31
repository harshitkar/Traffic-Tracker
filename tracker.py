import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.general import scale_boxes
from detector import load_model, get_dataloader, run_inference
from config import *

def init_deepsort():
    cfg = get_config()
    cfg.merge_from_file(CONFIG_DEEPSORT)
    return DeepSort(DEEPSORT_MODEL,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=torch.cuda.is_available())

def track():
    model, stride, names, device, imgsz = load_model()
    dataset = get_dataloader(SOURCE, imgsz, stride)
    deepsort = init_deepsort()

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if device.type != 'cpu' else im.float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = run_inference(model, im)

        for i, det in enumerate(pred):
            im0 = im0s[i] if isinstance(dataset, list) else im0s
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                xywhs = (det[:, :4]).cpu()
                confs = (det[:, 4]).cpu()
                clss = (det[:, 5]).cpu()
                outputs = deepsort.update(xywhs, confs, clss, im0)
            else:
                deepsort.increment_ages()
