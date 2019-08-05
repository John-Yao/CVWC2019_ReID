#!/usr/bin/env bash
# extract
DET_FNAME='../detections/work_dirs/enew2hard.rois'
TRACK4_IMGS_PATH='../detections/work_dirs/crop_images/enew2hard/'
DATASET_ROOT_DIR='/data/nif/tiger/reid/'
CUDA_VISIBLE_DEVICES=0 python sh_tiger/wild_aug_feat_extract.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK4_IMGS_DAPTH} \
                --det_fname  ${DET_FNAME} \
                --save_feats_fname 'enew2hard_augms.npy' \
                --fresh_feats --aug_ms \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/wild_aug_feat_extract.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK4_IMGS_DAPTH} \
                --det_fname  ${DET_FNAME} \
                --save_feats_fname 'enew2hard_augms.npy' \
                --fresh_feats --aug_ms \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/wild_aug_feat_extract.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK4_IMGS_DAPTH} \
                --det_fname  ${DET_FNAME} \
                --save_feats_fname 'enew2hard_augms.npy' \
                --fresh_feats --aug_ms \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center')" \

# extract
CUDA_VISIBLE_DEVICES=0 python sh_tiger/wild_aug_feat_extract.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK4_IMGS_DAPTH} \
                --det_fname  ${DET_FNAME} \
                --save_feats_fname 'enew2hard_augms.npy' \
                --fresh_feats --aug_ms \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/wild_aug_feat_extract.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK4_IMGS_DAPTH} \
                --det_fname  ${DET_FNAME} \
                --save_feats_fname 'enew2hard_augms.npy' \
                --fresh_feats --aug_ms \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/wild_aug_feat_extract.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK4_IMGS_DAPTH} \
                --det_fname  ${DET_FNAME} \
                --save_feats_fname 'enew2hard_augms.npy' \
                --fresh_feats --aug_ms \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center')" \
