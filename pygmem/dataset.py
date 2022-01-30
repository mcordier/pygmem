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
    def __init__(self, data, X, Z, group_ids, y, transform=None):
        """
        """
        self.grp_unique_ids = np.unique(group_ids, axis=1)
        self.data = data
        if isinstance(data, pd.DataFrame):
            # X, Z and Y needs to be list of columns or a column
            self.x = self.data[X].values
            self.z = [self.data[z].values for z in Z]
            self.y = self.data[y].values
        elif isinstance(data, np.array):
            # X, Z and Y needs to be list of integers
            self.x = self.data[:, X]
            self.z = [self.data[:, z] for z in Z]
            self.y = self.data[:, y]
        else:
            raise Exception('not good data type')
        self.y = torch.tensor(self.y,  dtype=torch.float32).unsqueeze(-1)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.z = [torch.tensor(z, dtype=torch.float32) for z in self.z]
        self.z = self.pad_tensors(self.z)
        self.z = torch.transpose(self.z, 0, 1)
        print(self.z.size())
        self.group_ids = torch.tensor(group_ids, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group_ids_idx = [self.group_ids[i,idx]
                    for i in range(len(self.group_ids))]
        sample = {'x': self.x[idx], 'y': self.y[idx],
                  'z': self.z[idx], 'group_ids': group_ids_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def pad_tensors(self, tensors):
        """
        Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

        The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
        where `Si` is the maximum value of dimension `i` amongst all tensors.
        """
        rep = tensors[0]
        padded_dim = []
        for dim in range(rep.dim()):
            max_dim = max([tensor.size(dim) for tensor in tensors])
            padded_dim.append(max_dim)
        padded_dim = [len(tensors)] + padded_dim
        padded_tensor = torch.zeros(padded_dim)
        padded_tensor = padded_tensor.type_as(rep)
        for i, tensor in enumerate(tensors):
            size = list(tensor.size())
            if len(size) == 1:
                padded_tensor[i, :size[0]] = tensor
            elif len(size) == 2:
                padded_tensor[i, :size[0], :size[1]] = tensor
            elif len(size) == 3:
                padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
            else:
                raise ValueError('Padding is supported for upto 3D tensors at max.')
        return padded_tensor


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
        self.grp_unique_ids = np.unique(group_ids)
        self.data = np.random.normal(-0.5, 0.2, size=(n_samples, n_features))
        self.x = self.data[:, X]
        self.z = self.data[:, Z]  # decide poisson theta values
        self.linear_f = np.random.randint(0, 1, size=len(X))-3
        self.linear_m = np.random.random(
            size=(len(Z), len(self.grp_unique_ids)))*2-1
        self.y = np.zeros((n_samples))
        self.y_hat = np.zeros((n_samples))

        for i in self.grp_unique_ids:
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

    def __init__(self, group_index, grp_unique_ids):
        self.group_index = group_index
        self.grp_unique_ids = grp_unique_ids

    def __iter__(self):
        random_sampling = np.random.permutation(self.grp_unique_ids)
        batch = []
        for k in random_sampling:
            batch = list(np.where(self.group_index == k))[0]
            yield batch


class BatchGroupSampler_fast(Sampler):
    '''
    Pytorch Sampler which create batches with the same userId
    (same index OID/OR/DEST). Use this one because it is faster.'''

    def __init__(self, group_index, grp_unique_ids):
        self.group_index = group_index
        self.grp_unique_ids = grp_unique_ids

    def __iter__(self):
        random_sampling = np.random.permutation(self.grp_unique_ids)
        batch = []
        for k in random_sampling:
            batch = self.group_index[k]
            yield batch

    def __len__(self):
        return(int(len(self.grp_unique_ids)))


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
    group_ids = np.array(dataset.group_ids)
    grp_unique_ids = np.array(dataset.grp_unique_ids)
    dataloaders_idx = []

    for i in range(len(grp_unique_ids)):
        # Create a list of the index (userId) of each value for the sampler
        group_index = np.empty((grp_unique_ids[i].max()+1,), dtype=object)
        # list of list of indexes per Ids_unique
        for (j, user) in enumerate(tqdm(group_ids[i])):
            if group_index[user] == None:
                group_index[user] = [j]
            else:
                group_index[user].append(j)

        # Sampler Creation
        sampler = BatchGroupSampler_fast(group_index, grp_unique_ids[i])
        # sampler = BatchGroupSampler(userIds_list, userIds_uniques)
        dataloaders_idx.append(DataLoader(dataset, batch_sampler=sampler,
                                    num_workers=0))
    return(dataloaders_idx)
