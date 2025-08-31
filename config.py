from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCE = "input.mp4"
YOLO_MODEL = "yolov5s.pt"
DEEPSORT_MODEL = "osnet_x0_25"
CONFIG_DEEPSORT = "deep_sort/configs/deep_sort.yaml"

OUT_DIR = ROOT / "inference/output"
PROJECT = ROOT / "runs/track"
NAME = "exp"

IMG_SIZE = [480]
CONF_THRES = 0.35
IOU_THRES = 0.5
MAX_DET = 1000

SHOW_VID = True
SAVE_VID = False
SAVE_TXT = False

DEVICE = ""
