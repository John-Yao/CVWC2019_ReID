# Solutions for CVWC2019 Track3&Track4 Challenge

### Note:
- Input size is 256x128
- Multi-scale test  is used.
- Multi-model ensemble is used. 
- The training of detection is not provided, we provide the detection results.
- Some of ReID model is pretrained on DeepFashion2. we only provide the pretrained model.
- Arcface loss is not used.
- We don't split the the data train and validation set.


## Environment
The code is developed using python 3.7 on Ubuntu 16.04.  NVIDIA GPUs are needed. The ReID models are developed and tested using 1 TITAN V GPU cards. The detection models are develop on 8xTesla V100. Other platforms or GPU cards are not fully tested.

## Quick start
Our project is based on the following repositories:

- (detection) https://github.com/open-mmlab/mmdetection
- (ReID) https://github.com/michuanhaohao/reid-strong-baseline

###  Track3 briefly description:

We build a ReID model base on global features and batch hard mining triplet loss. More details, some training tricks are used, including warmup,random erasing,centerloss and etc. For post-process, we use multi-scales augmentation and reranking.

### Track4 briefly description: 

The detection model is based on Cascade RCNN and coco finetune is used . The ReID model is trained with the same techique as Track3.

### Installation

1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).

2. Clone this repo, and we'll call the directory that you cloned as ${ReID_ROOT}.

3. Install dependencies following [repo1](https://github.com/michuanhaohao/reid-strong-baseline) and [repo2]( https://github.com/open-mmlab/mmdetection).

4. Install extra dependencies:
   ```
   pip install tqdm
   ```

5. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${ReID_ROOT}
    `-- model_checkpoint_pytorch
    	 `-- se_resnext50_32x4d-a260b3a4.pth
        `-- deepfashion2
            |-- Experiment4-senext50-224x224_lr2-bs8x8-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_center-arcface
            |   |-- se_resnext50_model_30.pth
            |   |-- se_resnext50_optimizer_30.pth
            |   |-- se_resnext50_optimizer_center_30.pth
   
   ```

### Data preparation
**For CVWC2019 Tiger ReID Track3&4 **, please download the data to ${DATASET_ROOT_DIR} , modify data_dir in *.py and modify DATASET_ROOT_DIR in *.sh ,  run the following command:

```shell
python prepare_train.py
python prepare_track3_crop_train.py
python prepare_track3_crop_test.py
python prepare_track4_test.py
DATASET_ROOT_DIR=/data/nif/tiger/reid/
cp ${DATASET_ROOT_DIR}/*.csv detections/work_dirs/crop_images/htc_reidtrain_e10_2200_1100/
cd reid-strong-baseline
# train
bash sh_tiger/train.sh
# merge feat
python merge_feat.py
# track3 test
bash sh_tiger/aug_test.sh
# track4 test
bash sh_tiger/wild_aug_test.sh
bash sh_tiget/wild_sub.sh
```

#### Testing on dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
download our trained model and put it in directory 'tiger/work_dirs/\*/\*' like this:
```
python prepare_train.py
# crop image a max detected area 
python prepare_track3_crop_train.py
python prepare_track3_crop_test.py
python prepare_track4_test.py
DATASET_ROOT_DIR=/data/nif/tiger/reid/
cp ${DATASET_ROOT_DIR}/*.csv detections/work_dirs/crop_images/htc_reidtrain_e10_2200_1100/
cd reid-strong-baseline

# merge feat
python merge_feat.py
# track3 test
bash sh_tiger/aug_test.sh
# track4 test
bash sh_tiger/wild_aug_test.sh
bash sh_tiget/wild_sub.sh
```



# Ablation Study

## track3

| model description                                         | mAP(single_cam) | mAP(cross_cam) |
| --------------------------------------------------------- | --------------- | -------------- |
| pcb_6 pooling vertically(softmax loss,not based the code) | 0.722           | 0.466          |
| senext50+alltricks(softmax + batch hard mining triplet)   | 0.782           | 0.487          |
| senext50+alltricks+multi-scale test+ reranking            | 0.841           | 0.524          |
| senext50+alltricks+multi-scale test+ reranking+crop image | 0.835           | 0.519          |
| 6 ensemble model+adjust the params of reranking           | 0.866           | 0.525          |

## track4

| Model description                                      | mAP(single_cam) | mAP(cross_cam) |
| ------------------------------------------------------ | --------------- | -------------- |
| senext50+alltricks                                     | 0.767           | 0.500          |
| senext50+alltricks+multi-scale test+ reranking         | 0.830           | 0.528          |
| senext50+alltricks+pretrain                            | 0.777           | 0.479          |
| senext50+alltricks+pretrain+multi-scale test+reranking | 0.828           | 0.510          |
| 6 ensemble model+adjust the params of reranking        | 0.841           | 0.539          |

## 
