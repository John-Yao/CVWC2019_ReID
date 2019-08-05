import numpy as np
save_dir = '../tiger/work_dirs/merge3/'
feat1 = np.load('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center/Exp6-se50-alltricks_center-test_feats_augms_cat.npy')
feat2 = np.load('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain/Exp7-se50-alltricks_center-test_feats_augms_cat.npy')
feat3 = np.load('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center/Exp8-se50-alltricks_center-test_feats_augms_cat.npy')
features = np.hstack([feat1,feat2,feat3])
os.makedirs(save_dir,exist_ok=True)
np.save(save_dir+'test_feats_augms_cat.npy',features)

save_dir = '../tiger/work_dirs/merge3/'
feat1 = np.load('../tiger/work_dirs/Experiment6-senext50-256x128-bs8x8-alltrics-triplet_center/enew2hard_augms_cat.npy')
feat2 = np.load('../tiger/work_dirs/Experiment7-senext50-alltricks-triplet_center-arcface-pretrain/enew2hard_augms_cat.npy')
feat3 = np.load('../tiger/work_dirs/Experiment8-senext50-256x128-bs8x8-alltricks-triplet_center/enew2hard_augms_cat.npy')
features = np.hstack([feat1,feat2,feat3])
os.makedirs(save_dir,exist_ok=True)
np.save(save_dir+'enew2hard_augms_cat.npy',features)

save_dir = '../tiger/work_dirs/merge3_1/'
feat1 = np.load('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center/Exp6-se50-alltricks_center-test_feats_augms_cat.npy')
feat2 = np.load('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain/Exp7-se50-alltricks_center-test_feats_augms_cat.npy')
feat3 = np.load('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center/Exp8-se50-alltricks_center-test_feats_augms_cat.npy')
features = np.hstack([feat1,feat2,feat3])
os.makedirs(save_dir,exist_ok=True)
np.save(save_dir+'test_feats_augms_cat.npy',features)

save_dir = '../tiger/work_dirs/merge3_1/'
feat1 = np.load('../tiger/work_dirs/Experiment6-senext50-256x128-bs16x4-alltrics-triplet_center/enew2hard_augms_cat.npy')
feat2 = np.load('../tiger/work_dirs/Experiment7-senext50-alltricks-bs16x4-triplet_center-arcface-pretrain/enew2hard_augms_cat.npy')
feat3 = np.load('../tiger/work_dirs/Experiment8-senext50-256x128-bs16x4-alltricks-triplet_center/enew2hard_augms_cat.npy')
features = np.hstack([feat1,feat2,feat3])
os.makedirs(save_dir,exist_ok=True)
np.save(save_dir+'enew2hard_augms_cat.npy',features)

# augms
save_dir = '../tiger/work_dirs/merge6/'
feat1 = np.load('../tiger/work_dirs/merge3/test_feats_augms_cat.npy')
feat2 = np.load('../tiger/work_dirs/merge3_1/test_feats_augms_cat.npy')
features = np.hstack([feat1,feat2])
os.makedirs(save_dir,exist_ok=True)
np.save(save_dir+'test_feats_augms_cat.npy',features)

save_dir = '../tiger/work_dirs/merge6/'
feat1 = np.load('../tiger/work_dirs/merge3/enew2hard_augms_cat.npy')
feat2 = np.load('../tiger/work_dirs/merge3_1/enew2hard_augms_cat.npy')

features = np.hstack([feat1,feat2])
os.makedirs(save_dir,exist_ok=True) 
np.save(save_dir+'enew2hard_augms_cat.npy',features)