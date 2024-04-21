# AlphaPose

[AlphaPose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the **first open-source system that achieves 70+ mAP (75 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset.** To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. 

<div align="center">
    <img src="media/AlphaPose_ihmc.gif", width="400"alt><br>
    <b><a href="https://github.com/Fang-Haoshu/Halpe-FullBody">Halpe 136 keypoints</a></b>
    <b><a href="https://www.youtube.com/watch?v=yn1EZ43aAk0">YouTube link</a></b><br>
</div>

## System Used
This system was tested on Ubuntu 20.04 with Nvidia RTX 3060 GPU, Cuda 11.3 and Nvidia Driver 535. 

## Setup
Clone the repo:
```
git clone https://github.com/ArghyaChatterjee/AlphaPose.git
```
Create a virtual environment:
```
cd AlphaPose/
python3 -m venv alpha_pose_venv
source alpha_pose_venv/bin/activate
pip3 install --upgrade pip
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install -r requirements.txt
echo "export PYTHONPATH=$(pwd):\${PYTHONPATH}" >> ~/.bashrc
source ~/.bashrc
sudo apt install libyaml-dev
python3 setup.py build develop
```
Install Pytorch 3D (Optional):
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip3 install git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable
```

## Download Pretrained Models
1. Download the object detection model manually: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place it into `detector/yolo/data`.
2. Or, if you want to use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) as the detector, you can download the weights [here](https://github.com/Megvii-BaseDetection/YOLOX), and place them into `detector/yolox/data`. We recommend [yolox-l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) and [yolox-x](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth).
3. Download the pose models and place them into `pretrained_models`. All models and details are available in the [Model Zoo](docs/MODEL_ZOO.md).

## Tracking 
Please read [trackers/README.md](trackers/) for details.

## Halpe Pose
### Inference Demo (Webcam) 
- Halpe 136 Keypoints Combined Pose Estimation
```
cd AlphaPose/ 
python3 scripts/alpha_pose_webcam_demo.py --cfg /home/arghya/alpha_pose/AlphaPose/configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml --checkpoint /home/arghya/alpha_pose/AlphaPose/pretrained_models/halpe136_fast50_dcn_combined_256x192_10handweight.pth --outdir examples/res --vis --webcam 0
```
- Halpe 136 Keypoints DCN Regression Pose Estimation
```
cd AlphaPose/ 
python3 scripts/alpha_pose_webcam_demo.py --cfg /home/arghya/alpha_pose/AlphaPose/configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml --checkpoint /home/arghya/alpha_pose/AlphaPose/pretrained_models/halpe136_fast50_dcn_regression_256x192.pth --outdir examples/res --vis --webcam 0
```
- Halpe 136 DUC Keypoints Regression Pose Estimation
```
cd AlphaPose/ 
python3 scripts/alpha_pose_webcam_demo.py --cfg /home/arghya/alpha_pose/AlphaPose/configs/halpe_136/resnet/configs/halpe_136/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint /home/arghya/alpha_pose/AlphaPose/pretrained_models/halpe136_fast152_duc_regression_256x192.pth --outdir examples/res --vis --webcam 0
```

## SMPL Pose 
### Colab Demo
[colab example](https://colab.research.google.com/drive/1_3Wxi4H3QGVC28snL3rHIoeMAwI2otMR?usp=sharing) 

### Inference Demo
``` bash
cd AlphaPose/
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME} # ${OUTPUT_DIR}, optional
```

Inference SMPL (Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) and put it in `model_files/`).
``` bash
./scripts/inference_3d.sh ./configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml ${CHECKPOINT} ${VIDEO_NAME} # ${OUTPUT_DIR}, optional
```
For high level API, please refer to `./scripts/demo_api.py`. To enable tracking, please refer to [this page](./trackers).

### Training 
Train from scratch:
``` bash
cd AlphaPose/
./scripts/train.sh ${CONFIG} ${EXP_ID}
```

Train `FastPose` on mscoco dataset.
``` bash
cd AlphaPose/
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```

### Validation 
Validate your model on MSCOCO val2017:
``` bash
cd AlphaPose/
./scripts/validate.sh ${CONFIG} ${CHECKPOINT}
```

## Fast Pose 
### Inference Demo (Image Files)

If you want to use `FastPose` model + `YOLOv3 SPP` as the detector:
``` bash
cd AlphaPose/
python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
```

If you want to use `FastPose` model + `yolox-x` as the detector:
```bash
python3 scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
```

### Inference Demo (Video Files)

If you want to use `FastPose` model + `YOLOv3 SPP` as the detector:
``` bash
./scripts/inference.sh configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml pretrained_models/fast_res50_256x192.pth ${VIDEO_NAME}
```

## Crowd Pose
Please read [docs/CrowdPose.md](docs/CrowdPose.md) for details.

## Detailed Demo
More detailed inference options and examples, please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md)
