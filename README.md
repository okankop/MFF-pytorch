# Motion Fused Frames (MFFs)

Pytorch implementation of the article [Motion fused frames: Data level fusion strategy for hand gesture recognition](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Kopuklu_Motion_Fused_Frames_CVPR_2018_paper.pdf) 

```diff
- Update: Code is updated for Pytorch 1.5.0 and CUDA 10.2
```

<p align="center"><img src="https://github.com/okankop/MFF-pytorch/blob/master/images/motion_fused_frames.jpg" align="middle" width="500" title="Motion Fused Frames" /></p>


### Installation
* Clone the repo with the following command:
```bash
git clone https://github.com/okankop/MFF-pytorch.git
```

* Setup in virtual environment and install the requirements:
```bash
conda create -n MFF python=3.7.4
pip install -r requirements.txt
```

### Dataset Preparation
Download the [jester dataset](https://www.twentybn.com/datasets/something-something) or [NVIDIA dynamic hand gestures dataset](http://research.nvidia.com/publication/online-detection-and-classification-dynamic-hand-gestures-recurrent-3d-convolutional) or [ChaLearn LAP IsoGD dataset](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html). Decompress them into the same folder and use [process_dataset.py](process_dataset.py) to generate the index files for train, val, and test split. Poperly set up the train, validatin, and category meta files in [datasets_video.py](datasets_video.py). Finally, use directory [flow_computation](https://github.com/okankop/flow_computation) to calculate the optical flow images using Brox method.

Assume the structure of data directories is the following:

```misc
~/MFF-pytorch/
   datasets/
      jester/
         rgb/
            .../ (directories of video samples)
                .../ (jpg color frames)
         flow/
            u/
               .../ (directories of video samples)
                  .../ (jpg optical-flow-u frames)
            v/
               .../ (directories of video samples)
                  .../ (jpg optical-flow-v frames)
    model/
       .../(saved models for the last checkpoint and best model)
```


### Running the Code
Followings are some examples for training under different scenarios:

* Train 4-segment network with 3 flow, 1 color frames (4-MFFs-3f1c architecture)
```bash
python main.py jester RGBFlow --arch BNInception --num_segments 4 \
--consensus_type MLP --num_motion 3  --batch-size 32
```

* Train resuming the last checkpoint (4-MFFs-3f1c architecture)
```bash
python main.py jester RGBFlow --resume=<path-to-last-checkpoint> --arch BNInception \
--consensus_type MLP --num_segments 4 --num_motion 3  --batch-size 32
```

* The command to test trained models (4-MFFs-3f1c architecture). Pretrained models are under [pretrained_models](pretrained_models).

```bash
python test_models.py jester RGBFlow pretrained_models/MFF_jester_RGBFlow_BNInception_segment4_3f1c_best.pth.tar --arch BNInception --consensus_type MLP --test_crops 1 --num_motion 3 --test_segments 4
```

All GPUs are used for the training. If you want a part of GPUs, use CUDA_VISIBLE_DEVICES=...

### Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@InProceedings{Kopuklu_2018_CVPR_Workshops,
author = {Kopuklu, Okan and Kose, Neslihan and Rigoll, Gerhard},
title = {Motion Fused Frames: Data Level Fusion Strategy for Hand Gesture Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```

### Acknowledgement
This project is built on top of the codebase [TSN-pytorch](https://github.com/yjxiong/temporal-segment-networks). We thank Yuanjun Xiong for releasing [TSN-Pytorch codebase](https://github.com/yjxiong/temporal-segment-networks), which we build our work on top. We also thank Bolei Zhou for the insprational work [Temporal Segment Networks](https://arxiv.org/pdf/1711.08496.pdf), from which we imported [process_dataset.py](https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py) to our project.
