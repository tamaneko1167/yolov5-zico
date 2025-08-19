# make_gamma_init.py
"""
ex:
python3 make_gamma_init.py \
  --cfg models/yolov5n.yaml \
  --weights yolov5n.pt \
  --gamma 1.5 \
  --out init_d033_w025_sppf_g150.pt
"""
import argparse, os, sys, yaml, torch, pathlib, urllib.request, tempfile, shutil
import torch.nn as nn

# yolov5 パスを通す（ローカル utils が使えるなら活用）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo import Model

# ---------------- utils: download / resolve weights ----------------
def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))

def _is_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))

def _download(url: str, dst: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    try:
        print(f"[download] {url} -> {dst}")
        urllib.request.urlretrieve(url, tmp)
        shutil.move(tmp, dst)
        return dst
    except Exception as e:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass
        raise RuntimeError(f"download failed: {url} ({e})")

def attempt_download_like_yolov5(name: str) -> str:
    """
    name がローカルに無ければ:
      1) utils.downloads.attempt_download があればそれを使う
      2) URLならそのままDL
      3) 'yolov5*.pt' なら GitHub Releases の既知タグから試行DL
    戻り値: ローカルの実ファイルパス
    """
    name = _expand(name)
    if os.path.isfile(name):
        return name

    # 1) 公式の downloader を使えるならそれを使う
    try:
        from utils.downloads import attempt_download as y5_attempt
        from utils.general import check_file as y5_check
        # check_file は存在しなければ attempt_download を呼ぶ実装
        p = y5_check(name)
        if not os.path.isfile(p):
            p = y5_attempt(name)
        if os.path.isfile(p):
            return p
    except Exception:
        pass  # フォールバックへ

    # 2) 直接URL指定
    if _is_url(name):
        dst = os.path.basename(name)
        return _download(name, dst)

    # 3) 既知の重み名（yolov5n/s/m/l/x など）を GitHub Releases から試す
    filename = os.path.basename(name)
    # 試すタグ（必要に応じて増やす）
    tags = ["v7.0", "v6.2", "v6.1", "v6.0"]
    urls = [f"https://github.com/ultralytics/yolov5/releases/download/{t}/{filename}" for t in tags]
    last_err = None
    for url in urls:
        try:
            return _download(url, filename)
        except Exception as e:
            last_err = e
            print(f"[warn] {e}")
    raise FileNotFoundError(
        f"weights '{name}' not found locally and download failed.\n"
        f"→ 手動でDLしてパスを渡すか、--weights '' でスクラッチにしてください。\n"
        f"last error: {last_err}"
    )

# ---------------- model helpers ----------------
def apply_bn_gamma(model: nn.Module, gamma: float):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                m.weight.mul_(float(gamma))

def load_state_dict_like_yolov5(model, weights_path):
    p = attempt_download_like_yolov5(weights_path)  # ← 既に使っているDL関数でOK
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)  # ★ココが肝
    except TypeError:
        # 古いTorch互換（weights_only引数が無い環境）
        ckpt = torch.load(p, map_location="cpu")

    # いろんな保存形式に対応
    if isinstance(ckpt, dict):
        if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            sd = ckpt["model"].state_dict()
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}（形変更による未ロード想定）")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}（不要キー）")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Bake BN gamma into initial weights (.pt)")
    ap.add_argument("--cfg", required=True, help="edited YAML (SPPF->none など反映済み)")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--gamma", type=float, default=1.0, help="BN scale to bake (default 1.0)")
    ap.add_argument("--weights", default=None, help="base weights: local path, URL, or yolov5n.pt")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    args = ap.parse_args()

    # 1) YAML（手編集済み）
    cfg_path = _expand(args.cfg)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # 2) モデル構築
    device = torch.device(args.device)
    model = Model(cfg).to(device).eval().float()

    # 3) 事前学習ロード（必要な場合）
    if args.weights and args.weights not in ("", "''"):
        load_state_dict_like_yolov5(model, args.weights)
    else:
        print("[info] no base weights provided → scratch init")

    # 4) BN γ を焼き込み
    if float(args.gamma) != 1.0:
        apply_bn_gamma(model, args.gamma)
        print(f"[info] applied BN gamma = {args.gamma}")
    else:
        print("[info] gamma=1.0（変更なし）")

    # 5) 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    ckpt = {"model": model, "epoch": -1}
    torch.save(ckpt, args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
