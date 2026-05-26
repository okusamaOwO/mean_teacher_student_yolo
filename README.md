# Mean Teacher Student YOLO

This repo contains a Mean Teacher (Student-Teacher) setup for YOLO models (mainly YOLOv9 architectures). It uses a dual/triple loss design paired with an Exponential Moving Average (EMA) teacher update. It's meant to help with semi-supervised or consistency learning for object detection.

## What's inside

- **Mean Teacher Loop**: The student model learns normally and via consistency loss, while we update the teacher smoothly using EMA.
- **YOLOv9**: Built on top of YOLOv9s.
- **Custom Losses**: We have `loss_tal_dual.py` and `loss_tal_triple.py` to handle the multiple outputs and consistency stuff.
- **Noise Injection**: We add noise (`helpers/add_noise.py`) to the student inputs so it doesn't just copy the teacher blindly.

## Setup

We're using `uv` for package management since it's much faster.

```bash
uv pip install -r requirements.txt
```

## How to use

### Data
Standard YOLO format. Just point to your images and labels in a YAML config (like `data1.yaml`).

### Training
Run the mean teacher training script. Tweak the batch size and workers depending on your GPU.

```bash
python train_dual_mean_teacher.py --workers 8 --device 0 --batch-size 16 --data data1.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9_mean_teacher --hyp data/hyps/hyp.scratch-high.yaml
```

### Validation
To test your trained model:

```bash
python val_dual.py --data data1.yaml --img 640 --device 0 --weights runs/train/yolov9_mean_teacher/weights/best.pt --name yolov9_val
```

## Notes
* [Colab Link](https://colab.research.google.com/drive/1uA6UVkMvJYnFFZjfwOlXRIN7tRPyXZ_P?authuser=4#scrollTo=NX1v-kdZByVn)
