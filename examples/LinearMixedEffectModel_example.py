'''
'''
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pygmem import gmem


nl_fun = None
n_samples = 1000
n_features = 2
X = [0, 1]
Z = [0]

group_ids = np.random.randint(10, size=n_samples)
grp_unique_ids = np.unique(group_ids)
data = np.random.normal(-0.5, 0.2, size=(n_samples, n_features))
x = data[:, X]
z = data[:, Z]  # decide poisson theta values
linear_f = np.random.randint(0, 1, size=len(X))-3
linear_m = np.random.random(size=(len(Z), len(grp_unique_ids)))*2-1
y = np.zeros((n_samples))
y_hat = np.zeros((n_samples))

for i in grp_unique_ids:
    if nl_fun is not None:
        y[np.where(group_ids == i)] = nl_fun(
            x[np.where(group_ids == i), :][0].dot(
                linear_f) + 1)
        y[np.where(group_ids == i)] += nl_fun(
            z[np.where(group_ids == i)][0].dot(
                linear_m[:, i]))
    else:
        y[np.where(group_ids == i)] = x[np.where(
            group_ids == i), :][0].dot(linear_f) + 1
        y[np.where(
            group_ids == i)] += z[np.where(
                group_ids == i)][0].dot(linear_m[:, i])
    y_hat[np.where(group_ids == i)] = np.exp(
        np.cos(y[np.where(group_ids == i)]))
    y[np.where(group_ids == i)] = np.random.poisson(
        y_hat[np.where(group_ids == i)])

df = pd.DataFrame(data, columns=["1", "2"])
df['y'] = y
df['y_bar'] = y_hat
df['id'] = group_ids
model = gmem.SimpleNonLinearMixedEffectModel() #, theta=torch.exp
model.fit(df, ["1", "2"], [['1']], [df.id.values], 'y',
          visualize=True, epochs=500)
df["y_hat"] = model.predict(df, ["1", "2"], [['1']], [df.id.values])

for i in grp_unique_ids:
    plt.scatter(df["1"].loc[np.where(group_ids == i)],
                df['y'].loc[np.where(group_ids == i)], label='y_true')
    plt.scatter(df["1"].loc[np.where(group_ids == i)],
                df['y_bar'].loc[np.where(group_ids == i)], label='y_bar')
    plt.scatter(df["1"].loc[np.where(group_ids == i)],
                df['y_hat'].loc[np.where(group_ids == i)], label='y_hat')
    plt.legend()
    plt.title(i)
    plt.show()