# Face Landmark & Gaze Estimation

Ckpt
---
- Gaze Estimation : [Gaze360.pkl](https://drive.google.com/file/d/1izHMezKvusUkNsosw1V3LkIPnc4phA1z/view?usp=sharing)
- Face Landmark  :  [spiga_wflw.pt](https://drive.google.com/uc?export=download&confirm=yes&id=1h0qA5ysKorpeDNRXe9oYkVcVe8UYyzP7)
- Face Detector  : Downloads run within code on first run

Requirements
---
- Linux
- Python < 3.11 
- PyTorch >= 1.10.1
- CUDA >= 11.0

Note
- This code has been tested for rtx 3060+ gpu.
- sort_tracker-py doesn't compatible with Python 3.11

Installation
---
```
pip install -r requirements.txt
```

When installing sort_tracker-py, use the following script if lap error occurs.

```
conda install -c conda-forge lap
```

Demo Flask
---
```
python face_ld_ge/demo_aimmo_flask.py --snapshot [/path/to/gaze/estimation/model/ckpt]
```

Demo Image
---
```
python face_ld_ge/demo_image.py --snapshot [/path/to/gaze/estimation/model/ckpt] --snapshot_landmark [/path/to/face/landmark/model/ckpt] --input [/path/to/input/directory] --output [/path/to/output/directory]
```

AI-Assist
---
```
python face_ld_ge/ai-assist.py --snapshot [/path/to/face/landmark/model/ckpt] --gpu [gpu_id] --input [/path/to/input/directory] --output [/path/to/output/directory]
```
