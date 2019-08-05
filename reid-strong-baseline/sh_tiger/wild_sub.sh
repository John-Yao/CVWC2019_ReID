#!/usr/bin/env bash
python sh_tiger/make_submission_reranking.py \
        --query_fname  '../detections/work_dirs/enew2hard.rois' \
        --query_feats_fname '../tiger/work_dirs/merge6/enew2hard_augms_cat.npy' \
        --save_json_fname '../tiger/work_dirs/merge6/enew2hard_augms_cat-reranking_k20_6_l025-sub.json'