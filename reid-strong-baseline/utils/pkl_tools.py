import os
import functools
from functools import wraps

try:
    import cPickle as pickle
except ImportError:
    import pickle


# def pkl_obj(pkl_key,pkl_dir,is_refresh):
#     def decorator(func):
#         @functools.wraps(func) ## 把func的__name__等属性复制到wrapper函数中
#         def wrapper(*args,**kwargs):
#             # [todo] smarter way to get pkl_key,pkl has to be passed by 'pkl_key=xx' in params list
#             pkl_file = kwargs.get(pkl_key, '')
#             cache_file = '.'.join([pkl_file, 'cache'])
#             cache_file = os.path.join(pkl_dir,cache_file)
#             print(cache_file)
#             if os.path.isfile(cache_file) and not is_refresh:
#                 with open(cache_file,'rb') as cache_fd:
#                     parse_result = pickle.load(cache_fd)
#             else:
#                 parse_result = func(*args, **kwargs)
#                 with open(cache_file, 'wb') as cache_fd:
#                     pickle.dump(parse_result, cache_fd)
#             return parse_result
#         return wrapper
#     return decorator
# @pkl_obj(pkl_key='id',pkl_dir='./',is_refresh=False)
# def test(x,obj_id=''):
#     return x
def pkl_obj(pkl_key,pkl_dir,is_refresh):
    def decorator(func):
        @functools.wraps(func) ## 把func的__name__等属性复制到wrapper函数中
        def wrapper(*args,**kwargs):
            # [todo] smarter way to get pkl_key,pkl has to be passed by 'pkl_key=xx' in params list
            pkl_file = kwargs.get(pkl_key, '')
            cache_file = '.'.join([pkl_file, 'cache'])
            cache_file = os.path.join(pkl_dir,cache_file)
            print(cache_file)
            if os.path.isfile(cache_file) and not is_refresh:
                with open(cache_file,'rb') as cache_fd:
                    parse_result = pickle.load(cache_fd)
            else:
                parse_result = func(*args, **kwargs)
                with open(cache_file, 'wb') as cache_fd:
                    pickle.dump(parse_result, cache_fd)
            return parse_result
        return wrapper
    return decorator
@pkl_obj(pkl_key='id',pkl_dir='./',is_refresh=False)
def test(x,obj_id=''):
    return x
if __name__ == "__main__":
    print(test([12,3],obj_id='pkl_obj.test'))