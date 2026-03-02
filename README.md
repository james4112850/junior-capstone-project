### 專案簡介

這個專案提供一個**車輛 ReID（Re-Identification）相似度比對**的簡易 Demo：輸入兩張圖片，先（可選）用 YOLO 偵測並裁切主要車輛，再用 ReID 模型抽特徵，計算兩張圖的 **cosine similarity**，並用 matplotlib 將結果並排顯示與輸出耗時統計。

### 主要功能

- **YOLO 車輛偵測/裁切（可選）**：使用根目錄的 `best.pt`，取面積最大的車輛框做裁切。
- **ReID 特徵抽取**：
  - **優先使用 FastReID 管線**（若 `fast-reid/` 相關依賴可正常匯入）。
  - 若 FastReID 不可用，會**自動回退**到簡化的 `torchvision.models.resnet50`（載入 checkpoint 內的 `model` 欄位為 state_dict 時會嘗試 `strict=False`）。
- **結果輸出**：顯示兩張圖片（原圖上畫出 YOLO 框與 conf）、cosine similarity，以及 YOLO / ReID 的分段耗時。

### 專案結構（重點）

- `test.py`：主要執行入口（讀圖 → YOLO 裁切 → ReID 抽特徵 → 相似度 → 視覺化）
- `best.pt`：YOLO 車輛偵測權重（**必須存在**才會啟用 YOLO；否則會報錯）
- `veri_deeplearning.pth`：ReID 權重（程式預設讀取此檔）
- `fast-reid/`：FastReID 原始碼（`test.py` 會將其加入 `sys.path` 以便匯入）
- `requirements.txt`：本專案最小可執行依賴（含 `ultralytics`）

### 環境需求

- **Python**：建議 3.10+（`test.py` 內已做 `collections` 相容處理）
- **PyTorch / CUDA**：依你的環境安裝對應版本；`requirements.txt` 內的 torch/torchvision 版本偏向 CUDA 11.8

### 安裝方式

在專案根目錄執行：

```bash
pip install -r requirements.txt
```

可選：若你想讓 `test.py` 優先走 FastReID 官方管線，可能還需要安裝 FastReID 依賴：

```bash
pip install -r fast-reid/docs/requirements.txt
```

> 備註：FastReID 依賴在不同環境可能還需要額外處理（例如 `faiss-gpu`、編譯等）。若匯入失敗，`test.py` 會自動回退到簡化 ResNet50 流程，仍可跑通整體比對。

### 權重檔放置

請確認以下檔案位於**專案根目錄**：

- `veri_deeplearning.pth`（ReID）
- `best.pt`（YOLO 車輛偵測）

### 使用方式

#### 方式一：指定兩張圖片路徑

```bash
python test.py path\to\img1.jpg path\to\img2.jpg
```

Windows（PowerShell）範例（使用專案內現成圖片）：

```bash
python .\test.py .\047.JPG .\047.JPEG
```

#### 方式二：不帶參數（自動挑資料夾內前兩張圖）

```bash
python test.py
```

程式會在 console 顯示：

- 是否成功載入 FastReID、是否啟用 YOLO
- YOLO 車輛框座標與 conf
- cosine similarity
- YOLO / ReID / 總耗時（毫秒）

並彈出 matplotlib 視窗：兩張原圖（畫框與 conf）+ 上方顯示 similarity。

### 常見問題

- **找不到 `best.pt`**
  - YOLO 初始化時會直接報錯：請把 YOLO 權重檔命名為 `best.pt` 並放在專案根目錄。
- **未安裝 `ultralytics`**
  - 會提示無法使用 YOLO，改用整張圖片做 ReID（不裁切）。
- **YOLO 沒偵測到車輛**
  - 會提示「YOLO 未偵測到車輛，改用整張圖片」，然後用整張圖做 ReID。
- **FastReID 匯入/載入失敗**
  - 會印出原因並回退到簡化 ResNet50 流程。
- **CUDA 不可用**
  - 會自動使用 CPU：`device = cuda if available else cpu`。

### 參考與授權

- `fast-reid/` 來自 FastReID 專案（Apache 2.0），詳見 `fast-reid/LICENSE`。
- 本專案內的模型權重（如 `best.pt`、`veri_deeplearning.pth`）之來源/授權請依你取得權重的原始來源為準。

