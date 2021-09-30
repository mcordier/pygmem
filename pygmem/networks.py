import torch
from torch.autograd import Variable
from torch.nn import functional as F


class LinearRegression(torch.nn.Module):
    '''
    Simple Linear Regression, for the fixed effects

    Args:
    ----------
    '''

    def __init__(self, n_X, n_y=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(n_X, n_y)
        torch.nn.init.normal_(self.linear.weight, std=1e-3)
        torch.nn.init.normal_(self.linear.bias, std=1e-3)

    def forward(self, x):
        return self.linear(x)


class LinearRandomEffects(torch.nn.Module):
    '''
    Linear Random Effect Model

    Args:
    ----------
        n_group : int
            number of individual coefficient to learn
    '''

    def __init__(self, n_Z, n_group):
        super(LinearRandomEffects, self).__init__()
        self.embeddings1 = torch.nn.Embedding(n_group, n_Z)
        torch.nn.init.normal_(self.embeddings1.weight, std=1e-7)
        self.embeddings2 = torch.nn.Embedding(n_group, 1)
        torch.nn.init.normal_(self.embeddings2.weight, std=1e-7)

    def forward(self, x, idx):
        a = self.embeddings1(idx)
        b = self.embeddings2(idx)
        rand_effect = torch.sum(x * a, dim=1, keepdim=True) + b
        return rand_effect


class LinearMixedEffects_fast(torch.nn.Module):
    '''
    Linear Mixed Effect Model, but suited for coordinated gradient descent.
    Indeed, we can put the parameters individually into different pytorch 
    optimizers, and then update the coefficient faster, thanks to the 
    ModuleDict (Dictionary of linear models). There is in this model, the 
    fixed effects AND the random effects.

    Args:
    ----------
        n_group : int
            number of individual coefficient to learn
    '''

    def __init__(self, n_X, n_Z, n_group, n_y=1):
        super(LinearMixedEffects_fast, self).__init__()
        self.linear_f = LinearRegression(n_X)
        print(n_Z, n_y, n_group)
        values = [torch.nn.Linear(n_Z, n_y) for i in range(n_group)]
        for val in values:
            torch.nn.init.normal_(val.weight, std=1e-7)
            torch.nn.init.normal_(val.bias, std=1e-7)
        keys = [str(i) for i in range(n_group)]
        self.linears = torch.nn.ModuleDict(dict(zip(keys, values)))

    def forward(self, x, z, idx):
        fixed_effect = self.linear_f(x)
        rand_effect = self.linears[idx](z)  # .view((1, -1))
        return fixed_effect + rand_effect


class BlockNet(torch.nn.Module):
    '''
    Linear Mixed Effect Model, but suited for coordinated gradient descent.
    Indeed, we can put the parameters individually into different pytorch 
    optimizers, and then update the coefficient faster, thanks to the ModuleDict
    (Dictionary of linear models). There is in this model, the fixed effects AND
    the random effects.

    Args:
    ----------
        n_group : int
            number of individual coefficient to learn
    '''

    def __init__(self, n_X, n_Z, n_group, n_y=1):
        super(BlockNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_X, 10)
        self.fc2 = torch.nn.Linear(10, 20)
        self.fc3 = torch.nn.Linear(20, 2)
        self.linear_mixed_effects = LinearMixedEffects_fast(2, 2, n_group, n_y)
        self.linear_f = self.linear_mixed_effects.linear_f
        self.linears = self.linear_mixed_effects.linears

    def forward(self, x, z, idx):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        fixed_effect = self.linear_f(x)
        rand_effect = self.linears[idx](x)  # .view((1, -1))
        return fixed_effect + rand_effect


class MemeNet(torch.nn.Module):
    '''
    Linear Mixed Effect Model, but suited for coordinated gradient descent.
    Indeed, we can put the parameters individually into different pytorch
    optimizers, and then update the coefficient faster, thanks to the ModuleDict
    (Dictionary of linear models). There is in this model, the fixed effects AND
    the random effects.

    Args:
    ----------
        n_group : int
            number of individual coefficient to learn
    '''

    def __init__(self, n_X, n_Z, n_group, enc_in=[1, 4], enc_out=[2, 30, 1]):
        super(MemeNet, self).__init__()
        # self.fc1 = torch.nn.Linear(n_X, n_X)

        self.fc_in = [torch.nn.Linear(enc_in[i], enc_in[i+1]) for i in range(
            len(enc_in)-1)]
        self.linear_mixed_effects = LinearMixedEffects_fast(enc_in[-1],
                                                            enc_in[-1],
                                                            n_group,
                                                            n_y=enc_out[0])
        self.fc_out = [torch.nn.Linear(
            enc_out[i], enc_out[i+1]) for i in range(len(enc_out)-1)]
        self.linear_f = self.linear_mixed_effects.linear_f
        self.linears = self.linear_mixed_effects.linears

    def forward(self, x, z, idx):

        for fci in self.fc_in:
            x = F.tanh(fci(x))

        x = self.linear_f(x) + self.linears[idx](x)
        for i, fci in enumerate(self.fc_out):
            if i == len(self.fc_in):
                x = fci(x)
            else:
                x = F.tanh(fci(x))

        return(x)


class MatrixFactorization(torch.nn.Module):
    '''Matrix Factorization model.

    Args:
    ----------
        n_users : int
            number of individual coefficient to learn (number of rows
            in the matrix)

        n_items : int 
            number of columns (number of departure date in the popularity 
            prediction problem

        n_factor : int
            Size of the vector representation. 
    '''

    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items,
                                               n_factors,
                                               sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)


class BiasedMatrixFactorization(torch.nn.Module):
    '''Matrix Factorization model but with a bias.

    Args:
    ----------
        n_users : int
            number of individual coefficient to learn (number of rows
            in the matrix)

        n_items : int 
            number of columns (number of departure date in the popularity 
            prediction problem

        n_factors : int
            Size of the vector representation. 
    '''

    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=True)
        torch.nn.init.normal_(self.user_factors.weight, std=1e-8)

        self.item_factors = torch.nn.Embedding(n_items,
                                               n_factors,
                                               sparse=True)
        torch.nn.init.normal_(self.item_factors.weight, std=1e-8)

        self.user_biases = torch.nn.Embedding(n_users,
                                              1,
                                              sparse=True)
        torch.nn.init.normal_(self.user_biases.weight, std=1e-8)

        self.item_biases = torch.nn.Embedding(n_items,
                                              1,
                                              sparse=True)
        torch.nn.init.normal_(self.item_biases.weight, std=1e-8)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) *
                 self.item_factors(item)).sum(dim=1, keepdim=True)
        return pred.squeeze()


# class LinearMixedEffect_MatrixFact(torch.nn.Module):
#     '''Linear Mixed Effect Model, but suited for coordinated gradient descent
#     and with matrix factorization terms. There is in this model,
#     the fixed effects
#     AND the random effects AND the matrix factorization.
#     Arg :
#         index_size (int) : number of individual coefficient
#                         to learnMatrix Factorization model but with a bias.
#         n_users (int) : number of individual coefficient to learn
#                         (number of rows in the matrix)
#         n_items (int) : number of columns (number of departure date
#                         in the popularity prediction problem
#         n_factor (int) : Size of the vector representation.
#     '''

#     def __init__(self, index_size, n_users, n_items, n_factors=20):
#         self.lmem = LinearMixedEffects_fast(index_size)
#         self.mf = BiasedMatrixFactorization(n_users, n_items, n_factors)

#     def forward(x, z, idx, user, item):
#         return(self.lmem(x, z, idx) + self.mf(user, item))
