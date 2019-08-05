#!/usr/bin/env bash
# Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center
# Experiment7-senext50-alltricks-triplet_center-arcface-pretrain
# Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center
# extract feat
TRACK3_IMGS_PATH='/share/nif/tiger/reid/test/'
DATASET_ROOT_DIR='/share/nif/tiger/reid/'
CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK3_IMGS_PATH} \
                --save_feats_fname 'Exp6-se50-alltricks_center-test_feats_augms.npy' \
                --save_json_fname 'Exp6-se50-alltricks_center-test_sub_augms_reranking.json' \
                --fresh_feats --aug_ms  \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK3_IMGS_PATH} \
                --save_feats_fname 'Exp7-se50-alltricks_center-test_feats_augms.npy' \
                --save_json_fname 'Exp7-se50-alltricks_center-test_sub_augms_reranking.json' \
                --fresh_feats --aug_ms  \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path '/share/yao/detections/reid/tiger/work_dirs/crop_images/htc_test_e10_reid_soft_flip/' \
                --save_feats_fname 'Exp8-se50-alltricks_center-test_feats_augms.npy' \
                --save_json_fname 'Exp8-se50-alltricks_center-test_sub_augms_reranking.json' \
                --fresh_feats --aug_ms  \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('../detections/work_dirs/crop_images/htc_reidtrain_e10_2200_1100/')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center')" \

# extract feat
CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK3_IMGS_PATH} \
                --save_feats_fname 'Exp6-se50-alltricks_center-test_feats_augms.npy' \
                --save_json_fname 'Exp6-se50-alltricks_center-test_sub_augms_reranking.json' \
                --fresh_feats --aug_ms  \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK3_IMGS_PATH} \
                --save_feats_fname 'Exp7-se50-alltricks_center-test_feats_augms.npy' \
                --save_json_fname 'Exp7-se50-alltricks_center-test_sub_augms_reranking.json' \
                --fresh_feats --aug_ms  \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain')" \

CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path '../detections/work_dirs/crop_images/htc_test_e10_reid_soft_flip/' \
                --save_feats_fname 'Exp8-se50-alltricks_center-test_feats_augms.npy' \
                --save_json_fname 'Exp8-se50-alltricks_center-test_sub_augms_reranking.json' \
                --fresh_feats --aug_ms  \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('../detections/work_dirs/crop_images/htc_reidtrain_e10_2200_1100/')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center')" \
# sub

CUDA_VISIBLE_DEVICES=0 python sh_tiger/aug_feat_extract_reranking.py --config_file='configs/tiger/softmax_triplet_with_center_fixaug.yml' \
                --imgs_path ${TRACK3_IMGS_PATH} \
                --save_feats_fname 'test_feats_augms.npy' \
                --save_json_fname 'test_feats_augms_cat_reranking_k20_4.json' \
                --aug_ms --sub \
                DATASETS.NAMES "('tiger')" \
                DATASETS.ROOT_DIR "('${DATASET_ROOT_DIR}')" \
                MODEL.DEVICE_ID "('0')" \
                MODEL.NAME "('se_resnext50')" \
                MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center/se_resnext50_model_120.pth')" \
                OUTPUT_DIR "('../tiger/work_dirs/merge6/')" \
