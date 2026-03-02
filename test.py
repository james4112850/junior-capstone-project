import sys
from pathlib import Path
import collections
import collections.abc
import time

import torch
from torch import nn
import numpy as np
import cv2  # fast-reid demo 也是用 cv2 做前處理
from matplotlib import pyplot as plt

# 為了相容 Python 3.10+，部分舊套件仍從 collections 匯入 Mapping 等型別
for _name in ["Mapping", "MutableMapping", "Sequence"]:
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# 將 fast-reid 加入匯入路徑（如果存在）
ROOT_DIR = Path(__file__).resolve().parent
FASTREID_DIR = ROOT_DIR / "fast-reid"
if FASTREID_DIR.is_dir():
    sys.path.insert(0, str(FASTREID_DIR))

# 嘗試匯入 FastReID
FASTREID_AVAILABLE = False
FASTREID_IMPORT_ERROR = None
try:
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultPredictor

    FASTREID_AVAILABLE = True
except Exception as e:
    FASTREID_AVAILABLE = False
    FASTREID_IMPORT_ERROR = e
    get_cfg = None
    DefaultPredictor = None

# 嘗試匯入 YOLO（ultralytics）
YOLO_AVAILABLE = False
YOLO_IMPORT_ERROR = None
_YOLO_MODEL = None
YOLO_CUSTOM_VEHICLE_MODEL = False  # 若使用 best.pt（專門車輛模型）則設為 True

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = e
    YOLO = None

from PIL import Image
from torchvision import models, transforms


def build_model_from_state_dict(state_dict: dict) -> nn.Module:
    """
    盡力根據 state_dict 建一個可用的 backbone。
    這裡假設是基於 ResNet-50 的 ReID 模型，最後特徵維度為 2048。
    如果你的專案裡有原始模型類別，建議改成直接匯入那個類別來載入權重。
    """
    # 嘗試使用 torchvision 的 resnet50 作為骨幹
    backbone = models.resnet50(weights=None)
    # 去掉分類頭，只保留全局特徵 (2048 維)
    backbone.fc = nn.Identity()

    # 嘗試載入 state_dict（忽略不匹配的鍵）
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[警告] 載入 state_dict 時存在不匹配的鍵：")
        if missing:
            print("  缺失鍵 (missing keys) 的數量:", len(missing))
        if unexpected:
            print("  多餘鍵 (unexpected keys) 的數量:", len(unexpected))

    return backbone


def get_preprocess():
    # ImageNet 通用預處理（備援用，未使用 fast-reid 時才會用到）
    return transforms.Compose(
        [
            transforms.Resize((256, 128)),  # ReID 常用尺寸 (H, W)，若有需要可自行修改
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _init_yolo_model():
    """延遲載入 YOLO 模型（若有安裝 ultralytics）。"""
    global _YOLO_MODEL, YOLO_CUSTOM_VEHICLE_MODEL
    if not YOLO_AVAILABLE:
        return None
    if _YOLO_MODEL is None:
        # 與資料前處理腳本一致：強制使用當前目錄下的 best.pt 作為車輛偵測模型
        best_path = ROOT_DIR / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到 YOLO 權重檔 best.pt，預期位置: {best_path}")

        print(f"[資訊] 載入自訂 YOLO 權重: {best_path.name} (專門偵測車輛)")
        _YOLO_MODEL = YOLO(str(best_path))
        YOLO_CUSTOM_VEHICLE_MODEL = True
    return _YOLO_MODEL


def detect_main_vehicle_bbox(img_bgr: np.ndarray):
    """
    使用 YOLO 偵測車輛，回傳面積最大的車輛框與其信度 (bbox, conf)：
      - bbox: (x1, y1, x2, y2)
      - conf: float
    偵測與座標處理邏輯，盡量與資料前處理腳本中「框車子」的方式保持一致：
      - model(image, conf=0.95, save=False, verbose=False)
      - 取 results[0].boxes.xyxy
      - 將座標裁切在影像範圍內，過濾掉無效框
    若未安裝 YOLO 或偵測失敗則回傳 None。
    """
    if not YOLO_AVAILABLE:
        return None

    model = _init_yolo_model()
    if model is None:
        return None

    # 與前處理腳本統一的 YOLO 呼叫方式
    results = model(img_bgr, conf=0.95, save=False, verbose=False)
    boxes_xyxy = results[0].boxes.xyxy if len(results) > 0 else []
    boxes_conf = results[0].boxes.conf if len(results) > 0 else None
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None

    h, w = img_bgr.shape[:2]
    best_idx = -1
    best_area = -1.0
    best_bbox = None
    best_conf = None

    # 遍歷所有框，做與前處理腳本相同的座標裁切，並取面積最大的一個
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        area = float((x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            best_idx = i
            best_bbox = (x1, y1, x2, y2)
            if boxes_conf is not None and len(boxes_conf) > i:
                try:
                    best_conf = float(boxes_conf[i].item())
                except Exception:
                    best_conf = None

    if best_idx < 0 or best_bbox is None:
        return None

    return best_bbox, best_conf


def crop_with_bbox(img_bgr: np.ndarray, bbox):
    if bbox is None:
        return img_bgr
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def load_image_and_detect_vehicle(img_path: Path):
    """
    讀取原圖（BGR），若 YOLO 可用則偵測主要車輛 bbox + conf，並回傳裁切車輛圖。

    Returns:
        img_bgr (np.ndarray): 原圖
        cropped_bgr (np.ndarray): 車輛裁切圖（若沒偵測到則等於原圖）
        bbox (tuple|None): (x1, y1, x2, y2)
        conf (float|None): bbox 的信度
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {img_path}")

    bbox = None
    conf = None
    if YOLO_AVAILABLE:
        try:
            det = detect_main_vehicle_bbox(img)
            if det is not None:
                bbox, conf = det
        except Exception as e:
            print(f"[警告] YOLO 偵測失敗，改用整張圖片。原因: {e}")
            bbox = None
            conf = None
    else:
        if YOLO_IMPORT_ERROR is not None:
            print(
                "[提示] 未安裝 ultralytics，無法使用 YOLO 偵測車輛，將直接使用整張圖片。\n"
                "       如需啟用，可先執行: pip install ultralytics"
            )

    if bbox is None:
        print("[提示] YOLO 未偵測到車輛，改用整張圖片。")
        return img, img, None, None

    x1, y1, x2, y2 = bbox
    conf_str = f"{conf:.3f}" if conf is not None else "N/A"
    print(f"[資訊] YOLO 車輛偵測框: ({x1}, {y1}), ({x2}, {y2})  conf={conf_str}")
    return img, crop_with_bbox(img, bbox), bbox, conf


def load_and_crop_vehicle(img_path: Path) -> np.ndarray:
    """
    讀取圖片，若 YOLO 可用則先偵測車輛並裁剪最大的車輛框；否則回傳整張圖。
    回傳值為 BGR 影像 (np.ndarray)。
    """
    # 保留舊介面：只回裁切圖（不含原圖/bbox/conf）
    _, cropped, _, _ = load_image_and_detect_vehicle(img_path)
    return cropped


def build_fastreid_predictor(ckpt_path: Path, device: torch.device):
    """
    使用 fast-reid 官方 config + DefaultPredictor 建立模型。
    對應 MODEL_ZOO 中的：
      configs/VeRi/sbs_R50-ibn.yml  <->  veri_sbs_R50-ibn.pth
    參考: MODEL_ZOO.md VeRi Baseline 區段。
    """
    if not FASTREID_AVAILABLE:
        raise RuntimeError("FastReID 未安裝或匯入失敗")

    cfg = get_cfg()
    config_file = FASTREID_DIR / "configs" / "VeRi" / "sbs_R50-ibn.yml"
    if not config_file.is_file():
        raise FileNotFoundError(f"找不到 FastReID VeRi config 檔案: {config_file}")

    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(ckpt_path)
    cfg.MODEL.DEVICE = str(device)

    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if not isinstance(ckpt, dict):
        # 直接是完整模型
        model = ckpt
    else:
        model_obj = ckpt.get("model", None)
        if model_obj is None:
            raise RuntimeError("checkpoint 中沒有 'model' 欄位")

        # 如果是完整模型物件
        if isinstance(model_obj, nn.Module):
            model = model_obj
        # 如果看起來像是 state_dict（一般就是 dict）
        elif isinstance(model_obj, dict):
            model = build_model_from_state_dict(model_obj)
        else:
            raise RuntimeError(
                f"無法識別的 'model' 類型: {type(model_obj)}，"
                "請確認 checkpoint 的儲存方式。"
            )

    model.to(device)
    model.eval()
    return model


def extract_feature_fastreid(
    cfg, predictor: DefaultPredictor, img_bgr: np.ndarray
) -> torch.Tensor:
    """
    使用 fast-reid 官方 DefaultPredictor 抽特徵，流程參考 demo/predictor.py + demo/demo.py：
      - BGR -> RGB
      - resize 到 cfg.INPUT.SIZE_TEST
      - 轉 tensor (B, C, H, W)
      - model forward
      - L2 normalize
    """
    import torch.nn.functional as F

    # BGR -> RGB
    img_rgb = img_bgr[:, :, ::-1]

    h, w = cfg.INPUT.SIZE_TEST
    img_resized = cv2.resize(
        img_rgb, (w, h), interpolation=cv2.INTER_CUBIC
    )  # 注意 cv2 的尺寸順序是 (W, H)

    image = torch.as_tensor(
        img_resized.astype("float32").transpose(2, 0, 1)
    )[None]  # [1, C, H, W]

    with torch.no_grad():
        feat = predictor(image)  # [1, D]

    feat = F.normalize(feat, dim=1)
    return feat.squeeze(0).cpu()


def extract_feature_torchvision(
    model: nn.Module, img_bgr: np.ndarray, device: torch.device
) -> torch.Tensor:
    preprocess = get_preprocess()
    # BGR -> RGB 再轉成 PIL Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    tensor = preprocess(img).unsqueeze(0).to(device)  # [1, C, H, W]

    with torch.no_grad():
        feat = model(tensor)

    # 有些 ReID 模型會回傳 (feat, logits) 或 dict
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    elif isinstance(feat, dict):
        # 嘗試幾個常見鍵名
        for key in ["feat", "features", "global_feat"]:
            if key in feat:
                feat = feat[key]
                break

    # [1, D] -> [D]
    feat = feat.view(feat.size(0), -1)
    feat = nn.functional.normalize(feat, p=2, dim=1)  # L2 normalize
    return feat.squeeze(0).cpu()


def cosine_similarity(f1: torch.Tensor, f2: torch.Tensor) -> float:
    f1 = f1 / (f1.norm(p=2) + 1e-12)
    f2 = f2 / (f2.norm(p=2) + 1e-12)
    return float(torch.dot(f1, f2).item())


def show_images_with_similarity(img1_bgr: np.ndarray, img2_bgr: np.ndarray, sim: float, title1: str, title2: str):
    """用 matplotlib 並排顯示兩張圖，並在最上方顯示 Cosine Similarity。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2)
    axes[1].axis("off")
    fig.suptitle(f"Cosine Similarity = {sim:.4f}", fontsize=14)
    plt.tight_layout()
    plt.show()


def draw_bbox_with_conf(img_bgr: np.ndarray, bbox, conf, color=(0, 0, 255)) -> np.ndarray:
    """在原圖上畫車輛 bbox（紅色）並在左上角寫 conf。"""
    out = img_bgr.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    text = f"{conf:.2f}" if conf is not None else "conf:N/A"
    # 文字改大很多，並加底框提升可讀性
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 4

    # 左上角文字（稍微往上移）；若太靠上則改放到框內上方
    tx = x1 + 8
    ty = y1 - 12
    if ty < 10:
        ty = y1 + 40

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # 底框：紅底
    pad_x, pad_y = 8, 6
    x0 = max(0, tx - pad_x)
    y0 = max(0, ty - th - pad_y)
    x1b = min(out.shape[1] - 1, tx + tw + pad_x)
    y1b = min(out.shape[0] - 1, ty + baseline + pad_y)
    cv2.rectangle(out, (x0, y0), (x1b, y1b), color, -1)
    # 白字
    cv2.putText(out, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def main():
    """
    使用方式:
        python test.py img1.jpg img2.jpg

    若未提供參數，預設使用:
        018.jpg 與 032.jpg
    """
    cwd = Path(__file__).resolve().parent

    if len(sys.argv) >= 3:
        img1_path = Path(sys.argv[1])
        img2_path = Path(sys.argv[2])
    else:
        # 未提供參數時，自動從目前目錄挑出兩張圖片
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_files = sorted(
            [p for p in cwd.iterdir() if p.is_file() and p.suffix.lower() in exts]
        )

        if len(img_files) < 2:
            raise FileNotFoundError(
                "目前資料夾中找不到足夠的圖片檔（至少需要 2 張，副檔名為 jpg/jpeg/png/bmp）。"
            )

        img1_path, img2_path = img_files[0], img_files[1]
        print(f"[提示] 未提供圖片路徑，預設使用 {img1_path.name} 與 {img2_path.name}")

    if not img1_path.is_file():
        raise FileNotFoundError(f"找不到圖片: {img1_path}")
    if not img2_path.is_file():
        raise FileNotFoundError(f"找不到圖片: {img2_path}")

    ckpt_path = cwd / "veri_deeplearning.pth"
    
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到模型權重檔: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[資訊] 使用裝置: {device}")

    use_fastreid = False
    cfg = None
    predictor = None

    if FASTREID_AVAILABLE:
        try:
            print("[資訊] 嘗試使用 FastReID 官方模型載入權重...")
            cfg, predictor = build_fastreid_predictor(ckpt_path, device)
            use_fastreid = True
            print("[資訊] 已啟用 FastReID 官方管線進行特徵抽取。")
        except Exception as e:
            print(f"[警告] FastReID 載入失敗，改用簡化 ResNet50 版本。原因: {e}")
    else:
        if FASTREID_IMPORT_ERROR is not None:
            print(
                "[警告] FastReID 套件匯入失敗，將改用簡化 ResNet50 版本。\n"
                f"       匯入錯誤: {FASTREID_IMPORT_ERROR}\n"
                "       建議到 fast-reid 專案根目錄安裝依賴，例如:\n"
                "         pip install -r fast-reid/docs/requirements.txt"
            )

    if not use_fastreid:
        print("[資訊] 載入簡化 ResNet50 模型中...")
        model = load_model(ckpt_path, device)

    # 計時：分開統計 YOLO 與 ReID 的時間
    # img1: YOLO
    t_yolo1_start = time.time()
    print(f"[資訊] 從 {img1_path} 抽取特徵（含 YOLO 車輛裁剪）...")
    img1_orig, img1_bgr, img1_bbox, img1_conf = load_image_and_detect_vehicle(img1_path)
    t_yolo1 = time.time() - t_yolo1_start

    # img1: ReID
    t_reid1_start = time.time()
    if use_fastreid:
        feat1 = extract_feature_fastreid(cfg, predictor, img1_bgr)
    else:
        feat1 = extract_feature_torchvision(model, img1_bgr, device)
    t_reid1 = time.time() - t_reid1_start

    # img2: YOLO
    t_yolo2_start = time.time()
    print(f"[資訊] 從 {img2_path} 抽取特徵（含 YOLO 車輛裁剪）...")
    img2_orig, img2_bgr, img2_bbox, img2_conf = load_image_and_detect_vehicle(img2_path)
    t_yolo2 = time.time() - t_yolo2_start

    # img2: ReID
    t_reid2_start = time.time()
    if use_fastreid:
        feat2 = extract_feature_fastreid(cfg, predictor, img2_bgr)
    else:
        feat2 = extract_feature_torchvision(model, img2_bgr, device)
    t_reid2 = time.time() - t_reid2_start

    sim = cosine_similarity(feat1, feat2)
    # 顯示並排圖片：原圖 + 紅框車輛 + conf（框左上角），並在上方顯示 cosine similarity
    vis1 = draw_bbox_with_conf(img1_orig, img1_bbox, img1_conf)
    vis2 = draw_bbox_with_conf(img2_orig, img2_bbox, img2_conf)
    show_images_with_similarity(vis1, vis2, sim, img1_path.name, img2_path.name)

    yolo_total = t_yolo1 + t_yolo2
    reid_total = t_reid1 + t_reid2
    total = yolo_total + reid_total

    print("\n================== ReID / YOLO 時間統計（單位：毫秒） ==================")
    print(f"- Cosine similarity: {sim:.4f}")

    print("\n- YOLO 偵測與裁剪")
    print(f"  - img1: {t_yolo1 * 1000:.1f} ms")
    print(f"  - img2: {t_yolo2 * 1000:.1f} ms")
    print(f"  - 小計: {yolo_total * 1000:.1f} ms")

    print("\n- ReID 特徵抽取")
    print(f"  - img1: {t_reid1 * 1000:.1f} ms")
    print(f"  - img2: {t_reid2 * 1000:.1f} ms")
    print(f"  - 小計: {reid_total * 1000:.1f} ms")

    print("\n- 整體處理時間")
    print(f"  - YOLO + ReID 總計: {total * 1000:.1f} ms")
    print("=====================================================================")


if __name__ == "__main__":
    main()


