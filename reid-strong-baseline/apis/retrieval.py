# ref: https://github.com/ihciah/deep-fashion-retrieval/blob/master/retrieval.py
import os
import sys
import pdb
import json
import cv2
import math
import glob
import time
import subprocess
from tqdm import tqdm
from PIL import Image
from multiprocessing import Process,Queue
from multiprocessing import Pool
import collections.abc as collections_abc

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.re_ranking import re_ranking
# unorder
def commom_process(func, data, nr_procs, *args):
    
    total = len(data)
    stride = math.ceil(total/nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    tqdm.monitor_interval = 0
    pbar = tqdm(total = total)

    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = data[start:end]
        p = Process(target= func,args=(result_queue, sample_data, *args))
        p.start()
        procs.append(p)

    for i in range(total):

        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update()
    for p in procs:
        p.join()
    return results
# https://github.com/open-mmlab/mmcv/blob/d99c6f8ddf67d1c98c60bb5967b7e007c20d718d/mmcv/utils/progressbar.py
def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)
def task_parallel(func,
                    tasks,
                    nproc,
                    initializer=None,
                    initargs=None,                    
                    chunksize=1,                       
                    keep_order=True):
    """Track the progress of parallel task execution with a progress bar.
    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.
    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
  
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.
    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)

    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)

    pool.close()
    pool.join()
    return results
def get_similarity(feature, feats, metric='cosine'):
    dist = cdist(np.expand_dims(feature, axis=0), feats, metric)[0]
    return dist
def get_top_n(dist, retrieval_top_n,query_id,remove_qid=False):
    if retrieval_top_n>=len(dist)-1:
        if remove_qid:
            ind = np.argsort(dist)[:retrieval_top_n+1]
            ind = [_x for _x in ind if _x != query_id][:retrieval_top_n]
        else:
            ind =  np.argsort(dist)[:retrieval_top_n]
        
        ret = list(zip(ind, dist[ind]))
        # ret = sorted(ret, key=lambda x: x[1], reverse=False) #
    else:
        if remove_qid:
            ind = np.argpartition(dist, retrieval_top_n+1)[:retrieval_top_n+1]
            ind = [_x for _x in ind if _x != query_id][:retrieval_top_n]
        else:
            ind = np.argpartition(dist, retrieval_top_n)[:retrieval_top_n]

        ret = list(zip(ind, dist[ind]))
        ret = sorted(ret, key=lambda x: x[1], reverse=False) #
    return {'query_id':query_id,'top_n':ret}
def naive_query(feature,query_id,feats,retrieval_top_n,remove_qid,metric='cosine'):
    # start_time = time.time()
    dist = cdist(np.expand_dims(feature, axis=0), feats, metric)[0]
    # print(time.time()-start_time)
    return get_top_n(dist,retrieval_top_n,query_id,remove_qid)
def retrieval_worker(result_queue,data,feats,retrieval_top_n,remove_qid,metric):
    for _data in data:
        result_queue.put_nowait(naive_query(_data[0],_data[1],feats,retrieval_top_n,remove_qid,metric))
def retrieval_reranking_gpu(qf,gf,retrieval_top_n,k1=20,k2=6,lambda_value=0.3,dist_metric = 'euclidean'):
    distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value,dist_metric=dist_metric)
    results = []
    for query_id,dist in enumerate(distmat):
        if retrieval_top_n>=len(dist)-1:
            ind =  np.argsort(dist)[:retrieval_top_n]
        else:
            ind = np.argpartition(dist, retrieval_top_n)[:retrieval_top_n]
        ret = list(zip(ind, dist[ind]))
        ret = sorted(ret, key=lambda x: x[1], reverse=False) #
        results.append( {'query_id':query_id,'top_n':ret})
    return results


# def retrieval_worker(result_queue,data,feats,retrieval_top_n,remove_qid,metric):
#     for _data in data:
#         start_time = time.time()
#         res = naive_query(_data[0],_data[1],feats,retrieval_top_n,remove_qid,metric)
#         print(time.time()-start_time)
#         result_queue.put_nowait(res)
# def retrieval_worker(result_queue,data,feats,retrieval_top_n,remove_qid,metric):
#     for _data in data:
#         feature,query_id = _data[0],_data[1]
#         dist = cdist(np.expand_dims(feature, axis=0), feats, metric)[0]
#         result_queue.put_nowait(get_top_n(dist,retrieval_top_n,query_id,remove_qid))

# def visualize(original, result, cols=1):
#     import matplotlib.pyplot as plt
#     import cv2
#     n_images = len(result) + 1
#     titles = ["Original"] + ["Score: {:.4f}".format(v) for k, v in result]
#     images = [original] + [k for k, v in result]
#     mod_full_path = lambda x: os.path.join(DATASET_BASE, x) \
#         if os.path.isfile(os.path.join(DATASET_BASE, x)) \
#         else os.path.join(DATASET_BASE, 'in_shop', x,)
#     images = list(map(mod_full_path, images))
#     images = list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), images))
#     fig = plt.figure()
#     for n, (image, title) in enumerate(zip(images, titles)):
#         a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
#         plt.imshow(image)
#         a.set_title(title)
#     fig.set_size_inches(np.array(fig.get_size_inches()) * n_images * 0.25)
#     plt.show()
if __name__ == '__main__':
    import time
    N_gallery = 1024
    N_test = 10
    feature = np.random.rand(1024) 
    feats   = np.random.rand(1024,1024) 
    start_time = time.time()
    for i in range(N_test):
        get_similarity(feature,feats)
    print((time.time()-start_time)/N_test)


    