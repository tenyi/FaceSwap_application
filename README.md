# Face

> **維護狀態說明**：原作者已停止維護，原始 `requirements.txt` 鎖定的是 2021 年的
> Python 3.6.12 + `tensorflow-gpu==1.14.0` + `torch==1.7.0`。這三個版本都**沒有
> Apple Silicon (arm64) 的 wheel**，在 M 系列 Mac 上無法安裝。專案本身的模型權重
> 與資料檔是齊全的，經過相容性修正後可在現代環境正常運作（詳見下方
> 「現代化安裝」）。

## 現代化安裝（實測於 macOS 26 / Apple Silicon）

```bash
# 全新 venv（會下載約 600 MB 的 torch + tensorflow）
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-modern.txt
python QTmainV2.py
```

若系統 Python 3.11 已經裝好 torch / tensorflow，可直接沿用、免下載：

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-modern.txt
python QTmainV2.py
```

首次執行卡通化功能時，`face_alignment` 會自動下載 SFD 人臉偵測器與 2DFAN 關鍵點
模型（約 180 MB）到 `~/.cache/torch/hub/checkpoints/`。

> **dlib 說明**：預設已改用純 PyTorch 的 SFD 偵測器，不需要 dlib。若你的環境中
> dlib 可正常載入，仍可傳 `detector='dlib'` 走原本的路徑。
>
> **Apple Silicon 注意事項**：`tensorflow` 在 macOS 只有 CPU 版本（原專案寫的
> `tensorflow-gpu==1.14.0` 只在 Linux/Windows 有 GPU 版），換臉運算在 M 晶片上
> 靠 CPU 即可。

### 效能

| 項目 | 修正前 | 修正後 |
| --- | --- | --- |
| 靜態圖換臉（1860×1840） | 17.6 s | **2.7 s** |
| 影片換臉（640×646） | 每影格約 20 s | **每影格 0.29 s** |
| 人像卡通化 | 15.2 s | 15 s |

主因是 `utils/render.py` 的 `render_texture` 原本是「87k 個三角形 × 每個像素」的
純 Python 三重迴圈，單次呼叫就要 7 秒。已改為分批向量化實作，**輸出逐像素完全
一致**（maxdiff = 0），單次呼叫降到 0.17 s。

### 著色模式

`PRNetfacetool(shading=...)`：

- `gouraud`（預設）— 三角形內用重心座標插值頂點顏色，消除原本「一塊三角形一個
  平色」的方格感
- `flat` — 完全沿用原始實作（每個三角形填單一平均色），輸出與原版**位元組相同**

```python
from PRNet_faceswap_tool import PRNetfacetool
tool = PRNetfacetool(shading='gouraud')   # 或 'flat'
```

### 已知限制

PRNet 用 256×256 的 UV 貼圖重建臉部，眼球區域的拓撲相當粗糙，因此**眼睛周圍在高
解析度下仍會有碎裂感**。這是 PRNet 模型本身的限制，不是實作問題；輸出解析度降低
（例如 640×646 的影片）時就不明顯。卡通化走的是 photo2cartoon 這套完全不同的模型，
沒有這個問題。

### 影片功能

```bash
# 影片偵測框 + 卡通化影片（輸出到 ./videos/）
python UseVideo.py input.mp4

# 單張照片卡通化
python cartoon.py --photo_path images/photo_test.jpg \
                  --save_path out.jpg --no-preview
```

GUI 的「選擇影片 / 選擇相片 / 執行換臉」走同一條路徑，處理時會在狀態列顯示
進度，再按一次「執行換臉」可中止。影片換臉輸出預設寫到
`./videos/PRNetvideo.mov`，目錄會自動建立。

### 相容性修正重點

| 檔案 | 問題 | 修正 |
| --- | --- | --- |
| `predictor.py` | 依賴 `tensorflow.contrib`（TF 2.0 已移除） | 以 `tf.compat.v1` 重建等價網路，變數名稱／形狀與原 checkpoint **完全一致**（245/245） |
| `utils/face_seg.py` | 使用 TF 1.x `Session` / `tf.ConfigProto` | 改用 `tf.compat.v1` + `tf.io.gfile` |
| `utils/face_detect.py` | `LandmarksType._2D` 在 face_alignment 1.3+ 改名 | 改用 `LandmarksType.TWO_D`（相容舊版 fallback） |
| `utils/preprocess.py` | 預設 `dlib` 偵測器 | 改預設 `sfd`（純 PyTorch，不需 dlib，偵測率也較好） |
| `utils/render.py` | `render_texture` 是純 Python 三重迴圈（7 s/次） | 分批向量化，**輸出逐像素一致**，0.17 s/次 |
| `utils/render.py` | 每個三角形只用 3 個頂點顏色的平均值填滿，放大後變成方格 | 新增 `shading='gouraud'`（預設）以重心座標插值；`shading='flat'` 可完全還原原始輸出（位元組相同） |
| `PRNet_faceswap_tool.py` | OpenCV 4.x 拒絕 numpy scalar 當座標 | 轉為 Python `float` |
| `PRNet_faceswap_tool.py` | 影片路徑缺 `videos/` 目錄、`cv2.destrouAllWindows` 拼字錯、未釋放 VideoWriter | 自動建目錄、修正拼字、`finally` 釋放資源 |
| `PRNet_faceswap_tool.py` | 影片每影格都重算參考臉 3D 重建 | 參考臉只算一次並重用（`build_reference`） |
| `PRNet_faceswap_tool.py` | 每影格重讀 eye mask PNG | 快取在實例上 |
| `PRNet_faceswap_tool.py` | `cv2.seamlessClone` 算了但結果被丟棄 | 移除（只浪費時間，且遮罩退化時會拋錯） |
| `QTmainV2.py` | 影片處理在 GUI 主執行緒，介面會凍結 | 移到 `QThread`，狀態列顯示進度，可中止 |
| `QTmainV2.py` | Qt 的 QThread 在 macOS 只有 **512 KB** 堆疊，`numpy.linalg.inv`（OpenBLAS）會 **Bus error** 崩潰 | 將 worker 堆疊設為 64 MB |
| `QTmainV2.py` | `regraph_trigger` 在 `start()` 之後才連線，結果可能遺失 | 改為啟動前連線且只連一次 |
| `QTmainV2.py` | 檔案對話框起點寫死 `"D:\\"` | 改用 home 目錄 |
| `cartoon.py` | 模組層級呼叫 `parse_args()`，任何帶參數的匯入都會被吃掉 | 移進 `__main__`，另加 `--no-preview` |
| `threeD_utilsv2.py` | `from cv2 import cv2` 在新版 OpenCV 已失效 | 移除該行 |
| `threeD_utilsv2.py` | `np.math` 在 numpy 1.25+ 已移除 | 改為 `np` |
| `utils/utils.py` | `scipy.misc.imsave` 在 SciPy 1.3+ 已移除 | 改用 Pillow |
| `predictor.py` | 同一 process 建立第二個實例會撞變數名稱 | 每個實例改用獨立 `tf.Graph` |

## 模型下載
[模型下載連結](https://drive.google.com/file/d/1lsQS8hOCquMFKJFhK_z-n03ixWGkjT2P/view) ( From : [photo2cartoon](https://github.com/minivision-ai/photo2cartoon) )

下載後會有 
**cartoon_data** 、 **photo2cartoon_weights.pt** 、 **seg_model_384.pb** 、 **model_mobilefacenet.pth** 

1.人像卡通化預訓練模型：photo2cartoon_weights.pt，存放在models路徑下。(20200504更新)

2.頭像分割模型：seg_model_384.pb，存放在utils路徑下。

3.人臉識別預訓練模型：model_mobilefacenet.pth，存放在models路徑下。

4.人像卡通化onnx模型：photo2cartoon_weights.onnx (Google雲端硬碟)，存放在models路徑下。

卡通化開源數據：cartoon_data，包含trainB和test。


[PRNet模型下載連結](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view)

5.下載後會有
( **256_256_resfcn256_weight.data-00000-of-00001** ) 檔案，將檔案移動至 Data/net-data 資料夾裡面。

PRNet Train Data:
https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view

https://github.com/YadiraF/PRNet/tree/master/Data

## 安裝環境
* **python==3.6.12**

* **pencv-python==4.1.0.25**

* **PyQt5==5.15.4**

* **matplotlib==3.3.3**

* **face-alignment==1.1.1**

* **tensorflow-gpu==1.14.0**

* **torchvision==0.8.1**

* **dlib==19.21.0**

## 執行檔案

```
python QTmainV2.py
```

### 初始畫面
![](./Doc/main.png)


## Reference

* 人像卡通化 ( Photo to Cartoon ) [[GitCode](https://github.com/minivision-ai/photo2cartoon)]
* Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network ( PRNet ) [[GitCode](https://github.com/YadiraF/PRNet)] [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf)]
* 技術參考 &rArr; ( [基于PRNet的3D人脸重建与替换](https://mp.weixin.qq.com/s?__biz=MzU1NzU2MzcyMw==&mid=2247483891&idx=1&sn=a0c658e02bb634d2ef3e1a94607537ca&chksm=fc32abd7cb4522c143d63ffd15e3950ffc2679995d076b9819052367f7428cca82bba52c897c&token=2041151839&lang=zh_CN#rd) )