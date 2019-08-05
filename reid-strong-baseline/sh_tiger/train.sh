# Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center
DATASET_ROOT_DIR=/data/nif/tiger/reid
python tools/train.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('tiger')" DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
    OUTPUT_DIR "('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center')" \
    MODEL.NAME "('se_resnext50')" \
    MODEL.PRETRAIN_CHOICE "('imagenet')" MODEL.PRETRAIN_PATH "('../model_checkpoint_pytorch/se_resnext50_32x4d-a260b3a4.pth')" 


python tools/train.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('tiger')" DATASETS.ROOT_DIR "('../detections/work_dirs/crop_images/htc_reidtrain_e10_2200_1100/')" \
    OUTPUT_DIR "('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center')" \
    MODEL.NAME "('se_resnext50')" \
    MODEL.PRETRAIN_CHOICE "('imagenet')" MODEL.PRETRAIN_PATH "('../model_checkpoint_pytorch/se_resnext50_32x4d-a260b3a4.pth')" 

python tools/train.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('tiger')" DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
    OUTPUT_DIR "('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain')" \
    MODEL.NAME "('se_resnext50')" \
    MODEL.PRETRAIN_CHOICE "('other')" MODEL.PRETRAIN_PATH "('../model_checkpoint_pytorch/deepfashion2/Experiment4-senext50-224x224_lr2-bs8x8-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_center-arcface/se_resnext50_model_30.pth')" 
    FRAMEWORK "('Baseline4')" MODEL.COSINE_LOSS_TYPE "('ArcFace')" MODEL.COSINE_LOSS_SCALE "(30.0)" MODEL.COSINE_LOSS_MARGIN "(0.50)" \
 

python tools/train.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('tiger')" DATASETS.ROOT_DIR "('../detections/work_dirs/crop_images/htc_reidtrain_e10_2200_1100/')" \
    OUTPUT_DIR "('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center')" \
    MODEL.NAME "('se_resnext50')" \
    MODEL.PRETRAIN_CHOICE "('imagenet')" MODEL.PRETRAIN_PATH "('../model_checkpoint_pytorch/se_resnext50_32x4d-a260b3a4.pth')" \
    DATALOADER.NUM_INSTANCE "(4)"

python tools/train.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('tiger')" DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
    OUTPUT_DIR "('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center')" \
    MODEL.NAME "('se_resnext50')" \
    MODEL.PRETRAIN_CHOICE "('imagenet')" MODEL.PRETRAIN_PATH "('../model_checkpoint_pytorch/se_resnext50_32x4d-a260b3a4.pth')"  \
    DATALOADER.NUM_INSTANCE "(4)"

python tools/train.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' MODEL.DEVICE_ID "('0')" \
     DATASETS.NAMES "('tiger')" DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
     OUTPUT_DIR "('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain')" \
     MODEL.NAME "('se_resnext50')" \
     MODEL.PRETRAIN_CHOICE "('other')" MODEL.PRETRAIN_PATH "('../model_checkpoint_pytorch/deepfashion2/Experiment4-senext50-224x224_lr2-bs8x8-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_center-arcface/se_resnext50_model_30.pth')" 
     FRAMEWORK "('Baseline4')" MODEL.COSINE_LOSS_TYPE "('ArcFace')" MODEL.COSINE_LOSS_SCALE "(30.0)" MODEL.COSINE_LOSS_MARGIN "(0.50)" \
     DATALOADER.NUM_INSTANCE "(4)"
