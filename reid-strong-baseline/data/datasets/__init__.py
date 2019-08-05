# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# # from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .tiger import Tiger,Tiger2
from .deepfashion2 import DeepFashion2,DeepFashion2_1
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    # # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'tiger':Tiger,
    'tiger2':Tiger2,
    'deepfashion2':DeepFashion2,
    'deepfashion2_1':DeepFashion2_1
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
