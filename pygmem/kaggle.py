import numpy as np
import torch
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
# from popularity_dataset import PopularityDataset
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import random
from dataset import Dataset_simulated, Dataset_gmem, BatchGroupSampler, BatchGroupSampler_fast
# from popularity_dataset import BatchGroupSampler, BatchGroupSampler_fast
from dataset import generate_dataloader_indexed
from utils import plot_data
from collections import OrderedDict
import networks
from gmem import GMEM


df_train = pd.read_csv('data/osic-pulmonary-fibrosis-progression/train.csv') #ratWeight
df_test = pd.read_csv('data/osic-pulmonary-fibrosis-progression/test.csv')
df_submission = pd.read_csv('data/osic-pulmonary-fibrosis-progression/sample_submission.csv', sep='_|,') #ratWeight
# print(df)
df_train['Patient'] = df_train['Patient'].astype('category')
# df = df.loc[0:13]
df_train[['id']]  = df_train[['Patient']].apply(lambda x: x.cat.codes)
map_code_cat = dict([(category, code) for code, category in enumerate(df_train.Patient.cat.categories)])
# dict(zip(df_train.Patient.cat.categories, df_train.Patient.cat.codes))

def mapping_cat_to_code(patient):
    return(map_code_cat[patient])

df_test['id'] = df_test['Patient'].apply(mapping_cat_to_code)
df_submission['id'] = df_submission['Patient'].apply(mapping_cat_to_code)
# # print(df_train['Patient'].value_counts())
# print(df_submission['Patient'].value_counts())
# print(df_submission['id'].value_counts())
# # print(df_submission['Patient'])
# n_unique_sub = df_submission['Patient'].unique()
# # print(df_submission['Patient'].unique())
# # print(map_code_cat)
# # print(map_code_cat[n_unique_sub[1]])
# print(df_train[df_train['Patient']== n_unique_sub[1]], map_code_cat[n_unique_sub[1]], df_submission[df_submission['Patient']==n_unique_sub[1]]['id'])
# print(map_code_cat)

# df['SmokingStatus'] = df['SmokingStatus'].astype('category')
# df[['SmokingStatus']]  = df[['SmokingStatus']].apply(lambda x: x.cat.codes)


# dataset = Dataset(df, ['time', 'weight'], ['time'], df.id.values, 'concentration')
model = GMEM(1, 1, max(np.unique(df_train.id.values))+1, enc_in=[], enc_out=[], theta=torch.exp) #, 

model.fit(df_train, ['Weeks'], ['Weeks'], 
        df_train.id.values, 'FVC', lr_f=10**(-2), 
        lr_r=0.001, epochs=1, visualize=False)

df_train['FVC_predicted'] = model.predict(df_train, ['Weeks'], ['Weeks'], df_train.id.values)
df_train['delta'] = np.abs(df_train['FVC_predicted']-df_train['FVC'])
df_train[df_train['delta']>1000]['delta'] = 1000
df_train['Confidence'] = np.sqrt(2) * df_train['delta']

confidence_mean_ids = df_train.groupby('id')['Confidence'].mean()
def conf_mean_ids_to_conf(ids):
    return(confidence_mean_ids.loc[ids])

df_test['FVC_predicted'] = model.predict(df_test, ['Weeks'], ['Weeks'], df_test.id.values)
df_test['delta'] = np.abs(df_test['FVC_predicted']-df_test['FVC'])
df_test[df_test['delta']>1000]['delta'] = 1000

df_test['Confidence'] = df_test['id'].apply(conf_mean_ids_to_conf)
# print((-np.sqrt(2)*df_test['delta']/df_test['Confidence']  - np.log(np.sqrt(2)*df_test['Confidence'] )).sum())

df_submission['FVC'] = model.predict(df_submission, ['Week'], ['Week'], df_submission.id.values)

df_submission['Confidence'] = df_submission['id'].apply(conf_mean_ids_to_conf)

cols = ['Patient', 'Week']
print(df_submission['Patient'].value_counts())
df_submission['Patient_Week'] = df_submission[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
df_submission = df_submission.drop(['Patient', 'Week', 'id', 'y'], axis=1)
df_submission = df_submission[['Patient_Week', 'FVC', 'Confidence']]
# print(df_submission)
# model = GMEM(1, 1, max(np.unique(df.id.values))+1) #, theta=torch.exp
# model.fit(df, ['week'], ['week'], df.id.values, 'weight')
