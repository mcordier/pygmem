import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


class Dataset_gmem(Dataset):
    """
    Customized Dataset class for pytorch data ingestion

    Args:
    ----------
        data : pandas.DataFrame or array-like

        X : [str]
            column names for fixed effects

        Z : [str]
            column names for fixed effects

        group_ids : [int]
            list of ids for the group index (no multi-groups)

        n_samples : int

        nl_fun : function

    Attributes
    ----------

    """
    def __init__(self, data, X, Z, groups, y, transform=None):
        """
        """
        self.unique_idx = np.unique(groups)
        self.data = data
        if isinstance(data, pd.DataFrame):
            # X, Z and Y needs to be list of columns or a column
            self.x = self.data[X].values
            self.z = self.data[Z].values
            self.y = self.data[y].values
        elif isinstance(data, np.array):
            # X, Z and Y needs to be list of integers
            self.x = self.data[:, X]
            self.z = self.data[:, Z]
            self.y = self.data[:, y]
        else:
            raise Exception('not good data type')
        self.y = torch.tensor(self.y,  dtype=torch.float32).unsqueeze(-1)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.z = torch.tensor(self.z, dtype=torch.float32)
        self.group_ids = torch.tensor(groups, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'x': self.x[idx], 'y': self.y[idx],
                  'z': self.z[idx], 'group_ids': self.group_ids[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Dataset_simulated(Dataset):
    """
    Customized Dataset class for simulated data (Poission) To be changed/erased
    soon.

    Args:
    ----------
        n_features : int

        X : int

        Z : int

        group_ids : list

        n_samples : int

        nl_fun : function
    """
    def __init__(
            self, n_features, X, Z, group_ids,
            n_samples=100, transform=None, nl_fun=None):
        """
        """
        self.group_index = np.unique(group_ids)
        self.data = np.random.normal(-0.5, 0.2, size=(n_samples, n_features))
        self.x = self.data[:, X]
        self.z = self.data[:, Z]  # decide poisson theta values
        self.linear_f = np.random.randint(0, 1, size=len(X))-3
        self.linear_m = np.random.random(
            size=(len(Z), len(self.group_index)))*2-1
        self.y = np.zeros((n_samples))
        self.y_hat = np.zeros((n_samples))

        for i in self.group_index:
            if nl_fun is not None:
                self.y[np.where(group_ids == i)] = nl_fun(
                    self.x[np.where(group_ids == i), :][0].dot(
                        self.linear_f) + 1)
                self.y[np.where(group_ids == i)] += nl_fun(
                    self.z[np.where(group_ids == i)][0].dot(
                        self.linear_m[:, i]))
            else:
                self.y[np.where(group_ids == i)] = self.x[np.where(
                    group_ids == i), :][0].dot(self.linear_f) + 1
                self.y[np.where(
                    group_ids == i)] += self.z[np.where(
                        group_ids == i)][0].dot(self.linear_m[:, i])
            self.y_hat[np.where(group_ids == i)] = np.exp(
                np.cos(self.y[np.where(group_ids == i)]))
            self.y[np.where(group_ids == i)] = np.random.poisson(
                self.y_hat[np.where(group_ids == i)])

        self.y = torch.tensor(self.y,  dtype=torch.float32).unsqueeze(-1)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.z = torch.tensor(self.z, dtype=torch.float32)
        self.group_ids = torch.tensor(group_ids, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'x': self.x[idx], 'y': self.y[idx],
                  'z': self.z[idx], 'group_ids': self.group_ids[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class BatchGroupSampler(Sampler):
    '''
    Pytorch Sampler which create batches with the same userId
    (same index OID/OR/DEST). Do not use this one, because it is
    slower.'''

    def __init__(self, userIds_list, userIds_uniques):
        self.userIds_list = userIds_list
        self.userIds_uniques = userIds_uniques

    def __iter__(self):
        random_sampling = np.random.permutation(self.userIds_uniques)
        batch = []
        for k in random_sampling:
            batch = list(np.where(self.userIds_list == k))[0]
            yield batch


class BatchGroupSampler_fast(Sampler):
    '''
    Pytorch Sampler which create batches with the same userId
    (same index OID/OR/DEST). Use this one because it is faster.'''

    def __init__(self, group_index, userIds_uniques):
        self.group_index = group_index
        self.userIds_uniques = userIds_uniques

    def __iter__(self):
        random_sampling = np.random.permutation(self.userIds_uniques)
        batch = []
        for k in random_sampling:
            batch = self.group_index[k]
            yield batch

    def __len__(self):
        return(int(len(self.userIds_uniques)))


def generate_dataloader_indexed(dataset):
    '''
    Generate a compatible fast pytorch dataloader from a pytorch
    popularity_dataset. The sampler constructs batches with same userIds
    (same rows, same OID/OR/DEST)

    Args:
    ----------
    dataset : Dataset

    returns:
    ----------
    dataloader_idx : Dataloader
        Dataloader for random effects with indexing in the right order
    '''

    # Train loaders by batch of same iDs
    userIds_list = np.array(dataset.group_ids)
    userIds_uniques = np.array(dataset.group_ids.unique())

    # Create a list of the index (userId) of each value for the sampler
    group_index = np.empty((userIds_uniques.max()+1,), dtype=object)
    # list of list of indexes per Ids_unique
    for (i, user) in enumerate(tqdm(userIds_list)):
        if group_index[user] == None:
            group_index[user] = [i]
        else:
            group_index[user].append(i)

    # Sampler Creation
    sampler = BatchGroupSampler_fast(group_index, userIds_uniques)
    # sampler = BatchGroupSampler(userIds_list, userIds_uniques)
    dataloader_idx = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=0)
    return(dataloader_idx)
