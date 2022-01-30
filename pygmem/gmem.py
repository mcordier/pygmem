import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pygmem import networks
from pygmem.dataset import generate_dataloader_indexed
from pygmem.dataset import Dataset_gmem


class GMEM():
    """
    Model main Class for Generalized Mixed Effect Models. This model have fit
    and predict function attributes similar to sklearn classes. It is
    initialized with the number of different variables in each categories, and
    the number of categories in each group of effects (n_groups). 

    Theta is the link function for generalized modelisation.

    By default, the model is linear, but it can be transformed into a 
    non-linear mixed effect model by setting the encoders "enc_in" and 
    "enc_out". The effect variables will be the vector of the in-between
    layer (mixed effect models in the new representation given by the non
    linear mapping)

    Args:
    ----------
        theta : function
            Link function for generalized models

        enc_in, enc_out : list[int]

        max_iterations: int

    Attributes
    ----------
        self.n_groups : int
            number of categories in the groups
        
        self.enc_in, self.enc_out : list[int]

        self.model : pygmem.wetworks
            Newtwork architecture to trained defined with enco_in and enc_out

        self.theta : function
            Link function for generalized models

        self.criterion : function
            Loss function used for the machine learning process

        self.max_iterations : int
            Maximum number of iterations for the optimization process
    """

    def __init__(
        self,
        theta=lambda x: x,
        enc_in=[], enc_out=[],
        max_iterations=1,
        criterion = torch.nn.MSELoss(reduction='sum')
    ):
        self.enc_in = enc_in
        self.enc_out = enc_out
        self.theta = theta
        self.criterion = criterion
        self.max_iterations = max_iterations
        self.model = None
        self.n_X = None
        self.n_Z = None
        self.n_groups = None

    def build_model(self, X, Z, group_ids):
        self.n_X = len(X)
        self.n_Z = [len(Z[i]) for i in range(len(Z))]
        self.n_groups = len(np.unique(group_ids, axis=0)[0])
        if (len(self.enc_in) + len(self.enc_out)) > 0:
            # self.model = networks.BlockNet(self.n_X, self.n_Z, n_groups)
            model = networks.MemeNet(self.n_X, self.n_Z, 
                self.n_groups, self.enc_in, self.enc_out)
        else:
            model = networks.LinearMixedEffects_fast(
                self.n_X, self.n_Z, self.n_groups)
        return(model)

    def predict(self, df, X, Z, group_ids):
        df['y_f'] = -1
        dataset = Dataset_gmem(df, X, Z, group_ids, 'y_f')
        train_dataloader = DataLoader(dataset, batch_size=100,
                                        shuffle=True, num_workers=0)
        running_loss = 0
        res = []
        for i_batch, sample_batched in enumerate(train_dataloader):
            x, z, y, group_id = sample_batched['x'], sample_batched[
                'z'], sample_batched['y'], sample_batched['group_ids'][0]
            print(sample_batched['group_ids'])
            eta_hat = self.model(x, z, group_id)
            y_pred = self.theta(eta_hat)
            print(y_pred.size())
            res+=y_pred[:, 0].tolist()
        return(np.array(res))

    def fit(
        self,
        df,
        X, Z,
        group_ids, y,
        X_val=None, Z_val=None,
        groups_val=None, y_val=None,
        lr_f=10**(-3), lr_r=0.001, epochs=2000, reset_weights=False,
        save_model=True, visualize=False
    ):
        """
        Fit gmem model.

        Agrs
        ----------
        df : {array-like, pd.DataFrame} of shape (n_samples, n_features)
            Training data.

        X : List of length self.n_X
            Training data columns or index names for fixed effects.

        Z : [List of length self.n_Z[]]
            Training data columns or index names for random effects.

        groups: {array-like, List}
            List of the group index for the first group (to change with multi-
            groups)

        y : {str, int}
            Name of the column or index of the predicted variable y

        lr_f : float
            Learning rate for fixed effects

        lr_r : float
            Learning rate for random effects

        epochs : int
            Number of epochs for each learning step (fixed and random effects).
            To be changed ?

        save_model : bool
            Save boolean to local data storage location

        visualize (temp): bool
            Enable visualition of the graphical output


        Returns
        -------
        """
        self.model = self.build_model(X, Z, group_ids)
        dataset = Dataset_gmem(df, X, Z, group_ids, y)
        self.dataset = dataset
        train_dataloader_f = DataLoader(dataset, batch_size=100,
                                        shuffle=True, num_workers=0)
        train_dataloaders_r = generate_dataloader_indexed(dataset)
        self.train(train_dataloader_f, train_dataloaders_r[0], lr_f, lr_r, epochs)
        if visualize:
            for i in dataset.grp_unique_ids[0][0:10]:
                plt.scatter(dataset.x[np.where(group_ids[0] == i), 0][0],
                            dataset.y[np.where(group_ids[0] == i)], label='y_true')
                plt.scatter(dataset.x[np.where(group_ids[0] == i), 0][0],
                    self.theta(self.model(dataset.x[np.where(group_ids[0] == i), :][0],
                        dataset.z[np.where(group_ids[0] == i), 0, :][0],
                        dataset.group_ids[0, np.where(group_ids[0] == i)][0]).detach()),
                    label='glmem')
                plt.legend()
                plt.show()
        if save_model:
            if not os.path.exists('model_dir'):
                os.makedirs('model_dir')
            self.save('model_dir')

    def train(self, train_dataloader_f, train_dataloader_r, lr_f, lr_r, epochs):
        """
        Training process of gmem

        Args
        ----------
        train_dataloader_f : DataLoader
            DataLoader for the fixed effect dataset

        train_dataloader_r : DataLoader
            DataLoader for the random effect dataset

        lr_f : float
            Learning rate for fixed effects

        lr_r : float
            Learning rate for random effects

        epochs : int
            Number of epochs for each learning step (fixed and random effects).
            To be changed ?

        Returns
        -------
        """
        if len(self.enc_in) + len(self.enc_out) == 0:
            optimizer_f = torch.optim.Adam(
                self.model.linear_f.parameters(), lr=lr_f)
            # One optimizer for each linear model
            optimizer_r = [torch.optim.Adam(
                self.model.linears[str(i)].parameters(), lr=lr_r)
                           for i in tqdm(range(self.n_groups))]
        else:

            optimizer_f = list(self.model.linear_f.parameters())
            for fci in self.model.fc_in:
                optimizer_f += list(fci.parameters())
            for fci in self.model.fc_out:
                optimizer_f += list(fci.parameters())
            optimizer_f = torch.optim.Adam(optimizer_f, lr=lr_f)

            # One optimizer for each linear model
            optimizer_r = [torch.optim.Adam(
                self.model.linears[str(i)].parameters(), lr=lr_r)
                           for i in tqdm(range(self.n_groups))]
        for i in range(self.max_iterations):
            print('1/2 : Fixed Effects')
            self.train_f(train_dataloader_f, epochs, optimizer_f)
            print('2/2 : Random Effects')
            self.train_r(train_dataloader_r, epochs, optimizer_r)

    def train_f(self, train_dataloader, epochs, optimizer):
        """
        Trains the pytorch model fixed effects.
        The random effects are NOT computed nor updated.

        Args
        ----------
        train_dataloader : DataLoader
            DataLoader for the fixed effect dataset

        epochs : int
            Number of epochs for each learning step (fixed and random effects).
            To be changed ?

        optimizer : torch.optim
            Optimizer for fixed effect trainings

        Returns
        -------
        """

        iters = tqdm(range(epochs))
        for iteration in iters:
            running_loss = 0.0
            # Forward pass. Te batch samples should be from the same userId.
            for i_batch, sample_batched in enumerate(train_dataloader):
                x, z, y, group_id = sample_batched['x'], sample_batched[
                    'z'], sample_batched['y'], sample_batched['group_ids'][0]
                #print(group_id)
                optimizer.zero_grad()
                # eta_hat = []
                # for i in range(len(group_id)):
                #     eta_hat.append(self.model(x[i], z[i], group_id[i]))
                # eta_hat = torch.cat(eta_hat)
                # eta_hat = eta_hat.reshape((len(x), 1))
                eta_hat = self.model(x, z, group_id)
                y_pred = self.theta(eta_hat)
                loss = self.criterion(y_pred, y)
                # print("--------")
                # print(y.size(), y_pred.size())
                # print("--------")
                eta_hat.backward(1.0/len(sample_batched) * (y_pred - y)) # Backward pass
                # loss.backward()
                optimizer.step()
                running_loss += loss.item()  # *len(sample_batched)
            iters.set_postfix_str(s=str(running_loss/len(train_dataloader.dataset)), refresh=True)
        # self.model.synchronize_linears()

    def train_r(self, train_dataloader, epochs, optimizer):
        """
        Trains the pytorch model random effects.

        Args
        ----------
        train_dataloader : DataLoader
            DataLoader for the random effect dataset

        epochs : int
            Number of epochs for each learning step (fixed and random effects).
            To be changed ?

        optimizer : [torch.optim]
            List of Optimizer for each category for random effect training

        Returns
        -------
        """
        iters = tqdm(range(epochs))
        for iteration in iters:
            running_loss = 0.0
            # Forward pass. Te batch samples should be from the same userId.
            for i_batch, sample_batched in enumerate(train_dataloader):
                x, z, y, group_id = sample_batched['x'], sample_batched[
                    'z'], sample_batched['y'], sample_batched['group_ids'][0]
                #print(z.size)
                optimizer[group_id[0]].zero_grad()
                eta_hat = self.model(x, z, str(group_id[0].item()), embedding=False)
                y_pred = self.theta(eta_hat)
                loss = self.criterion(y_pred, y)  # Backward pass
                eta_hat.backward(1.0/len(sample_batched) * (y_pred - y))
                # loss.backward()
                optimizer[group_id[0]].step()
                running_loss += loss.item()*len(sample_batched)
            iters.set_postfix_str(s=str(running_loss/len(train_dataloader.dataset)), refresh=True)
        self.model.synchronize_linear_r()
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


class GaussianLinearMixedEffectModel(GMEM):
    """
    Model main Class for Generalized Mixed Effect Models. This model have fit
    and predict function attributes similar to sklearn classes. It is
    initialized with the number of different variables in each categories, and
    the number of categories in each group of effects (n_groups). 

    Theta is the link function for generalized modelisation.

    By default, the model is linear, but it can be transformed into a 
    non-linear mixed effect model by setting the encoders "enc_in" and 
    "enc_out". The effect variables will be the vector of the in-between
    layer (mixed effect models in the new representation given by the non
    linear mapping)

    Args:
    ----------
        theta : function
            Link function for generalized models

        enc_in, enc_out : list[int]

        max_iterations: int

    Attributes
    ----------
        self.n_groups : int
            number of categories in the groups
        
        self.enc_in, self.enc_out : list[int]

        self.model : pygmem.wetworks
            Newtwork architecture to trained defined with enco_in and enc_out

        self.theta : function
            Link function for generalized models

        self.criterion : function
            Loss function used for the machine learning process

        self.max_iterations : int
            Maximum number of iterations for the optimization process
    """

    def __init__(
        self,
        max_iterations=1,
        criterion = torch.nn.MSELoss(reduction='sum')
    ):
        super().__init__()
        self.theta=lambda x: x
        self.enc_in=[]
        self.enc_out=[]

class PoissonLinearMixedEffectModel(GMEM):
    """
    Model main Class for Generalized Mixed Effect Models. This model have fit
    and predict function attributes similar to sklearn classes. It is
    initialized with the number of different variables in each categories, and
    the number of categories in each group of effects (n_groups). 

    Theta is the link function for generalized modelisation.

    By default, the model is linear, but it can be transformed into a 
    non-linear mixed effect model by setting the encoders "enc_in" and 
    "enc_out". The effect variables will be the vector of the in-between
    layer (mixed effect models in the new representation given by the non
    linear mapping)

    Args:
    ----------
        theta : function
            Link function for generalized models

        enc_in, enc_out : list[int]

        max_iterations: int

    Attributes
    ----------
        self.n_groups : int
            number of categories in the groups
        
        self.enc_in, self.enc_out : list[int]

        self.model : pygmem.wetworks
            Newtwork architecture to trained defined with enco_in and enc_out

        self.theta : function
            Link function for generalized models

        self.criterion : function
            Loss function used for the machine learning process

        self.max_iterations : int
            Maximum number of iterations for the optimization process
    """

    def __init__(
        self,
        max_iterations=1,
        criterion = torch.nn.MSELoss(reduction='sum')
    ):
        super().__init__()
        self.theta=torch.exp
        self.enc_in=[]
        self.enc_out=[]

class LogitLinearMixedEffectModel(GMEM):
    """
    Model main Class for Generalized Mixed Effect Models. This model have fit
    and predict function attributes similar to sklearn classes. It is
    initialized with the number of different variables in each categories, and
    the number of categories in each group of effects (n_groups). 

    Theta is the link function for generalized modelisation.

    By default, the model is linear, but it can be transformed into a 
    non-linear mixed effect model by setting the encoders "enc_in" and 
    "enc_out". The effect variables will be the vector of the in-between
    layer (mixed effect models in the new representation given by the non
    linear mapping)

    Args:
    ----------
        theta : function
            Link function for generalized models

        enc_in, enc_out : list[int]

        max_iterations: int

    Attributes
    ----------
        self.n_groups : int
            number of categories in the groups
        
        self.enc_in, self.enc_out : list[int]

        self.model : pygmem.wetworks
            Newtwork architecture to trained defined with enco_in and enc_out

        self.theta : function
            Link function for generalized models

        self.criterion : function
            Loss function used for the machine learning process

        self.max_iterations : int
            Maximum number of iterations for the optimization process
    """

    def __init__(
        self,
        max_iterations=1,
        criterion = torch.nn.MSELoss(reduction='sum')
    ):
        super().__init__()
        self.theta=torch.logit
        self.enc_in=[]
        self.enc_out=[]

class SimpleNonLinearMixedEffectModel(GMEM):
    """
    Model main Class for Generalized Mixed Effect Models. This model have fit
    and predict function attributes similar to sklearn classes. It is
    initialized with the number of different variables in each categories, and
    the number of categories in each group of effects (n_groups). 

    Theta is the link function for generalized modelisation.

    By default, the model is linear, but it can be transformed into a 
    non-linear mixed effect model by setting the encoders "enc_in" and 
    "enc_out". The effect variables will be the vector of the in-between
    layer (mixed effect models in the new representation given by the non
    linear mapping)

    Args:
    ----------
        theta : function
            Link function for generalized models

        enc_in, enc_out : list[int]

        max_iterations: int

    Attributes
    ----------
        self.n_groups : int
            number of categories in the groups
        
        self.enc_in, self.enc_out : list[int]

        self.model : pygmem.wetworks
            Newtwork architecture to trained defined with enco_in and enc_out

        self.theta : function
            Link function for generalized models

        self.criterion : function
            Loss function used for the machine learning process

        self.max_iterations : int
            Maximum number of iterations for the optimization process
    """

    def __init__(
        self,
        theta = lambda x: x,
        max_iterations=1,
        criterion = torch.nn.MSELoss(reduction='sum')
    ):
        super().__init__()

    def build_model(self, X, Z, group_ids):
        self.n_X = len(X)
        self.n_Z = None
        self.n_groups = len(np.unique(group_ids, axis=0)[0])
        self.enc_in=[self.n_X, 4, 8]
        self.enc_out=[8, 4, 2, 1]
        if (len(self.enc_in) + len(self.enc_out)) > 0:
            # self.model = networks.BlockNet(self.n_X, self.n_Z, n_groups)
            model = networks.MemeNet(self.n_X, self.n_Z, 
                self.n_groups, self.enc_in, self.enc_out)
        else:
            model = networks.LinearMixedEffects_fast(
                self.n_X, self.n_Z, self.n_groups)
        return(model)