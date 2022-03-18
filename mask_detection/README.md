# Mask detection - INRIA Project

## Mask detection on frame

### Dataset

You can download the dataset [here](https://www.kaggle.com/andrewmvd/face-mask-detection).
After running '**python pre_process.py**' dataset will be extracted in the a new folder **datasets/yolov5** at the root of the project.

### 🚀&nbsp; Installation

- Refer to principal README, section installation.

### 🧑🏻‍💻&nbsp; Train

- To train face mask detection:

```bash
python pre_process.py
python YOLOv5/train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 100 --data YOLOv5/data/mask_data.yaml --weights yolov5s.pt --workers 24 --name yolo_mask_det
```

### 🧑🏻‍💻&nbsp; Test

- To test face mask detection on test images:

```bash
python YOLOv5/detect.py --source image_path --weights yolov5s.pt --conf 0.25 --name yolo_road_det
```
