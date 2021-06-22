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


groups = np.random.randint(0, 3, size=1000)
# model = GMEM(n_X=1, n_Z=1, n_groups=3, nbr_layers=2)
df = pd.read_csv('data/theophyllineData.csv')
print(max(np.unique(df.id.values)))
model = GMEM(2, 1, max(np.unique(df.id.values))) #, theta=torch.exp
model.fit(df, ['time', 'weight'], ['time'], df.id.values-1, 'concentration', visualize=True)


df = pd.read_csv('data/theophyllineData.csv') #ratWeight
print(df)
df['id'] = df['id'].astype('category')
# df = df.loc[0:13]
df[['id']]  = df[['id']].apply(lambda x: x.cat.codes)

# dataset = Dataset(df, ['time', 'weight'], ['time'], df.id.values, 'concentration')
model = GMEM(1, 1, max(np.unique(df.id.values))+1, enc_in=[1, 4], enc_out=[4, 30, 1]) #, theta=torch.exp
model.fit(df, ['time'], ['time'], df.id.values, 'concentration', 
    lr_f=0.005, visualize=True)
print(model.predict(df, ['time'], ['time'], df.id.values))
# model = GMEM(1, 1, max(np.unique(df.id.values))+1) #, theta=torch.exp
# model.fit(df, ['week'], ['week'], df.id.values, 'weight')
