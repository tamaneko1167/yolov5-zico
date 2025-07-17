# test_strip.py

from utils.general import strip_optimizer

pt_file = 'runs/train/yolov5n_baseline/weights/best.pt'

strip_optimizer(pt_file)