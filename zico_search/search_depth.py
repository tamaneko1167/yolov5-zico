# zico_search/search_depth.py

import yaml
import torch
from pathlib import Path
from models.yolo import Model
from search_yolo.score import compute_zico_score, get_flops_and_params

def build_model_from_yaml(yaml_path, depth_multiple, width_multiple=0.5):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg['depth_multiple'] = depth_multiple
    cfg['width_multiple'] = width_multiple
    return Model(cfg, ch=[3])  # 入力チャンネル3（RGB）

def main():
    depths = [0.33, 0.5, 0.66, 1.0]
    results = []

    for d in depths:
        print(f"\n🔍 Evaluating depth_multiple = {d}")
        model = build_model_from_yaml('models/yolov5n.yaml', d)
        x = torch.randn(1, 3, 640, 640)
        zico = compute_zico_score(model, x)
        flops, params = get_flops_and_params(model, x)

        results.append({'depth': d, 'ZiCo': zico, 'FLOPs': flops, 'Params': params})

    # 表形式で出力
    import pandas as pd
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('search_yolo/results.csv', index=False)

if __name__ == '__main__':
    main()
