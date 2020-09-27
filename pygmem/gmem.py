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

class GMEM(object):
    """
    """
    def __init__(
        self,
        n_X,
        n_Z,
        n_groups,
        theta=lambda x:x,
        enc_in=[], enc_out=[],
        max_iterations=1
    ):
        self.n_groups = n_groups
        self.enc_in = enc_in
        self.enc_out = enc_out

        if (len(self.enc_in) + len(self.enc_out))>0:
            # self.model = networks.BlockNet(n_X, n_Z, n_groups)
            self.model = networks.MemeNet(n_X, n_Z, n_groups, enc_in, enc_out)
        else:
            self.model = networks.LinearMixedEffects_fast(n_X, n_Z, n_groups)

        self.theta = theta
        self.criterion = torch.nn.MSELoss()
        self.theta= theta #torch.exp
        self.max_iterations = max_iterations
        # self.nbr_layers = nbr_layers

    def predict(self, df, X, Z, groups):
        df['y'] = -1
        dataset = Dataset_gmem(df, X, Z, groups, 'y')
        train_dataloader = generate_dataloader_indexed(dataset)
        running_loss = 0
        res = []
        for i_batch, sample_batched in enumerate(dataset):
            # print(i_batch)
            x, z, y, group_id = sample_batched['x'], sample_batched[
                'z'], sample_batched['y'], sample_batched['group_ids']
            # print(group_id)
            eta_hat = self.model(x, z, str(group_id.item())) #self.model.linear_f(x)
            y_pred = self.theta(eta_hat)
            res.append(y_pred[0].item())
            loss = self.criterion(y_pred, y)
            running_loss += loss.item() #*len(sample_batched)
        print('running_loss:' + str(running_loss))
        return(np.array(res))

    def fit(
        self,
        df,
        X, Z,
        groups, y,
        X_val = None, Z_val = None,
        groups_val = None, y_val = None,
        lr_f=10**(-3), lr_r=0.001, epochs=2000, reset_weights=False,
        save_model=True, visualize=False
    ):
        # n_features = 2
        # dataset = Dataset_simulated(n_features, X, Z, groups, n_samples=1000, transform=None, nl_fun=None)#
        dataset = Dataset_gmem(df, X, Z, groups, y)
        self.dataset=dataset
        train_dataloader_f = DataLoader(dataset, batch_size=100,
                        shuffle=True, num_workers=0)
        train_dataloader_r = generate_dataloader_indexed(dataset)
        self.train(train_dataloader_f, train_dataloader_r, lr_f, lr_r,epochs)
        # print(self.dataset.linear_f, self.dataset.linear_m)
        # print([param.data for param in self.model.parameters()])
        if visualize:
            for i in dataset.unique_idx:
                plt.scatter(dataset.x[np.where(groups==i), 0], dataset.y[np.where(groups==i)], label='y_true')
                # plt.scatter(dataset.x[np.where(groups==i), 0], dataset.y_hat[np.where(groups==i)], label='y_hat_true')
                plt.scatter(dataset.x[np.where(groups==i), 0], self.theta(self.model(dataset.x[np.where(groups==i), :], 
                    dataset.z[np.where(groups==i), :], str(i)).detach()), label='glmem')
                # print(self.model.linear_f(dataset.x).detach().size())
                # plt.scatter(dataset.x[np.where(groups==i), 0],
                #             self.theta(self.model.linear_f(dataset.x[np.where(groups==i), :])).detach(),
                #             label='linear')
                plt.legend()
                plt.show()
        if save_model:
            if not os.path.exists('model_dir'):
                os.makedirs('model_dir')
            self.save('model_dir')


    def train(self, train_dataloader_f, train_dataloader_r, lr_f, lr_r, epochs):
        if len(self.enc_in) + len(self.enc_out)==0:
            optimizer_f = torch.optim.Adam(self.model.linear_f.parameters(), lr=lr_f)
            # One optimizer for each linear model
            optimizer_r = [torch.optim.Adam(self.model.linears[str(i)].parameters(), lr=lr_r)
                                for i in tqdm(range(self.n_groups))]
        else:

            optimizer_f = list(self.model.linear_f.parameters())
            for fci in self.model.fc_in:
                optimizer_f += list(fci.parameters())
            for fci in self.model.fc_out:
                optimizer_f += list(fci.parameters())
            optimizer_f = torch.optim.Adam(optimizer_f, lr=lr_f)

            # One optimizer for each linear model
            optimizer_r = [torch.optim.Adam(self.model.linears[str(i)].parameters(), lr=lr_r)
                                for i in tqdm(range(self.n_groups))]
        for i in range(self.max_iterations):
            print('1/2 : Fixed Effects')
            self.train_f(train_dataloader_f, epochs, optimizer_f)
            print('2/2 : Random Effects')
            self.train_r(train_dataloader_r, epochs, optimizer_r)

    def train_f(self, train_dataloader, epochs, optimizer):
        """
        Train the pytorch model fixed effects. The random effects are NOT computed nor
        updated.
        arg : 
        - train_dataloader (pytorch dataloader) : dataloader of the popularity dataset

        - lr (float) : learning rate for the gradient descent

        - epochs (int): number of epochs to train

        - limits (int): if the number of data point in a batch (in a row) is lower than this limit,
        then the individual coefficient corresponding to this batch should not be trained. (Ex :
        Only one point data do not return a good fitted model.)

        - reset_weights (Bool) : NOT IMPLEMENTED. If activated, then renitialized all the weights of 
        the model.
        """
         # One optimizer for each linear model
        # torch.nn.init.uniform_(self.model.linear_f.linear.weight, int(self.dataset.linear_f)-0.001, int(self.dataset.linear_f)+0.001)
        # torch.nn.init.uniform_(self.model.linear_f.linear.bias, 1-0.001, 1+0.001)
        iters = tqdm(range(epochs))
        for iteration in iters:
            # print(self.dataset.linear_f, self.dataset.linear_m)
            # print(self.model.parameters().data)

            # for param in self.model.parameters():
            #     print(param.data)
            #     break
            running_loss = 0.0
            # Forward pass. Te batch samples should be from the same userId.
            for i_batch, sample_batched in enumerate(train_dataloader):
                # print(i_batch)
                x, z, y, group_id = sample_batched['x'], sample_batched[
                    'z'], sample_batched['y'], sample_batched['group_ids']
                optimizer.zero_grad()
                eta_hat = self.model(x, z, str(group_id[0].item())) #self.model.linear_f(x)
                y_pred = self.theta(eta_hat)
                loss = self.criterion(y_pred, y) # Backward pass
                eta_hat.backward(1.0/len(sample_batched) * (y_pred - y))
                # loss.backward()
                optimizer.step()
                running_loss += loss.item() #*len(sample_batched)
            iters.set_postfix_str(s=str(running_loss), refresh=True)

    def train_r(self, train_dataloader, epochs, optimizer):
        """
         Train the pytorch model. The training is by batch, where each batch should be 
        from the same index (OID/OR/DEST). This should be implemented in the dataloader. 
        arg : 
        - train_dataloader (pytorch dataloader) : dataloader of the popularity dataset

        - lr (float) : learning rate for the gradient descent

        - epochs (int): number of epochs to train

        - limits (int): if the number of data point in a batch (in a row) is lower than this limit,
        then the individual coefficient corresponding to this batch should not be trained. (Ex :
        Only one point data do not return a good fitted model.)

        - reset_weights (Bool) : NOT IMPLEMENTED. If activated, then renitialized all the weights of 
        the model.
        """
        iters = tqdm(range(epochs))
        for iteration in iters:
            running_loss = 0.0
            # print(self.dataset.linear_f, self.dataset.linear_m)
            # for param in self.model.parameters():
            #     print(param.data)
            # Forward pass. Te batch samples should be from the same userId.
            for i_batch, sample_batched in enumerate(train_dataloader):
                x, z, y, group_id = sample_batched['x'], sample_batched[
                    'z'], sample_batched['y'], sample_batched['group_ids']
                optimizer[group_id[0]].zero_grad()
                eta_hat = self.model(x, z, str(group_id[0].item()))
                y_pred = self.theta(eta_hat)
                loss = self.criterion(y_pred, y) # Backward pass
                eta_hat.backward(1.0/len(sample_batched) * (y_pred - y))
                # loss.backward()
                optimizer[group_id[0]].step()
                running_loss += loss.item()*len(sample_batched)
            iters.set_postfix_str(s=str(running_loss), refresh=True)

    @staticmethod
    def get_model_path(model_dir, step=0):
        """
        Get model path
        :param model_dir:
        :param step:
        :return: path
        """
        basename = 'Glmem'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.model.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        """
        Save trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.model.state_dict(), model_path)

# groups = np.random.randint(0, 3, size=1000)
# # model = GMEM(n_X=1, n_Z=1, n_groups=3, nbr_layers=2)
# df = pd.read_csv('data/theophyllineData.csv')
# print(max(np.unique(df.id.values)))
# model = GMEM(2, 1, max(np.unique(df.id.values))) #, theta=torch.exp
# model.fit(df, ['time', 'weight'], ['time'], df.id.values-1, 'concentration')


# df = pd.read_csv('data/theophyllineData.csv') #ratWeight
# print(df)
# df['id'] = df['id'].astype('category')
# # df = df.loc[0:13]
# df[['id']]  = df[['id']].apply(lambda x: x.cat.codes)

# # dataset = Dataset(df, ['time', 'weight'], ['time'], df.id.values, 'concentration')
# model = GMEM(1, 1, max(np.unique(df.id.values))+1, enc_in=[1, 4], enc_out=[4, 30, 1]) #, theta=torch.exp
# model.fit(df, ['time'], ['time'], df.id.values, 'concentration', lr_f=0.005)
# print(model.predict(df, ['time'], ['time'], df.id.values))
# model = GMEM(1, 1, max(np.unique(df.id.values))+1) #, theta=torch.exp
# model.fit(df, ['week'], ['week'], df.id.values, 'weight')
