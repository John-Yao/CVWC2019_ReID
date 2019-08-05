import os
import json
import glob
import pdb
import cv2

import numpy as np
import pandas as pd 
if __name__ == '__main__':
    if 1:
        # the root directory of track3's reid dataset
        data_dir = '/data/nif/tiger/'
        data_dir = data_dir+'reid/'
        
        np.random.seed(491001)
        df = pd.read_csv(data_dir+'reid_list_train.csv',header=None,sep=',')
        df = df.rename(columns={0:'id',1:'filename'})

        identities = list(set(df['id']))
        identities = sorted(identities)
        np.random.shuffle(identities)
        split_train_rate = 0.85
        train_identities = identities[:int(len(identities)*split_train_rate)]
        valid_identities = identities[int(len(identities)*split_train_rate):]
        df_train = df[df['id'].isin(train_identities)]
        df_valid = df[df['id'].isin(valid_identities)]

        # map identity to start from 0 and increment 1
        train_identity_map = {identity:idx for idx,identity in enumerate(set(train_identities))}
        valid_identity_map = {identity:idx for idx,identity in enumerate(set(valid_identities))}
        df_train['id'] = df_train['id'].apply(lambda x: train_identity_map[x])
        df_valid['id'] = df_valid['id'].apply(lambda x: valid_identity_map[x])

        df_train.to_csv(data_dir+'list_train.csv',index=False,columns=['id', 'filename'],header=None)
        df_valid.to_csv(data_dir+'list_valid.csv',index=False,columns=['id', 'filename'],header=None)
        # get query and gallery
        columns = ['id','filename']
        query_outputs = {col:[] for col in columns}
        gallery_outputs = {col:[] for col in columns}
        #
        valid_identities = {}
        for _,row in df_valid.iterrows():
            if row['id'] not in valid_identities.keys():
                valid_identities[row['id']] = []
            valid_identities[row['id']].append(row['filename'])
        # split:
        for identity,filenames in valid_identities.items():
            filenames = sorted(filenames)
            np.random.shuffle(filenames)
            for i in range(len(filenames)):
                assert len(filenames)*0.3>1
                if i < len(filenames)*0.3:
                    query_outputs['id'].append(identity)
                    query_outputs['filename'].append(filenames[i])
                else:
                    gallery_outputs['id'].append(identity)
                    gallery_outputs['filename'].append(filenames[i])
        pd.DataFrame(query_outputs).to_csv(data_dir+'list_query.csv',index=False,columns=['id', 'filename'],header=None)
        pd.DataFrame(gallery_outputs).to_csv(data_dir+'list_gallery.csv',index=False,columns=['id', 'filename'],header=None)
    if 1:
        df_train = pd.read_csv(data_dir+'reid_list_train.csv',header=None).rename(columns={0:'id',1:'filename'})
        train_identities = df_train['id'].tolist()
        train_identity_map = {identity:idx for idx,identity in enumerate(set(train_identities))}        
        df_train['id'] = df_train['id'].apply(lambda x: train_identity_map[x])
        df_train.to_csv(data_dir+'list_alltrain.csv',index=False,columns=['id', 'filename'],header=None)
        
        




        