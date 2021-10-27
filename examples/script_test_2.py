"""
==========================================================================
ratWeight Example: Introduction to Generalized Mixed Effect Models
==========================================================================

In general, Mixed Effect Models try to learn fixed effect parameters, like in 
a regular regression, and some random effect parameters derived from a group
structure. There are a vector parameter per group category to learn, which makes
usually the computation of those parameters more difficult. From a statistical
point of view, we could be also interested in the variance of the parameters 
regarding the group population, but in our context, we only focus on prediction
and machine learning task, so we will only be interested in a fast and realiable
prediction from ou models.

This example illustrates both linear and non linear generalized mixed effect
models in the example of ratWeight dataset. The dependant variable is
the weight and the independent variables are the number of weeks.
Each rat is unique and are identified with the ids. Then, it is difficult 
to get a general prediction and that is where the GMEM becomes interesting. 
We will use the default link function (identity) because this is a regression
(continuous domain)

First, we print for 2 group categories the evolution of concentration  with time
and illustrates the individualisation of prediction.

Then, we fit a linear mixed effect model with 1 fixed variables (week)
and 1 random effect variable (week). We see that the linear hypothesis is not
suited.

Then, we fit a non linear mixed effect model with 1 fixed variables (week)
and 1 random effect variable (week). Here, with only variable, we try to predict
individualized predictions. We set an encoder architecture for vectors
transformation and the random and fixed effect are learned in the layer of the
middle, which enables the maximum of flexibility (low bias) in the learning pro-
cess.

"""
print(__doc__)



import numpy as np
import pandas as pd
import torch
from pygmem.gmem import GMEM

df = pd.read_csv('data/ratWeight.csv')
df['id'] = df['id'].astype('category')
df[['id']]  = df[['id']].apply(lambda x: x.cat.codes)

# Linear Model
model = GMEM(1, 1, max(np.unique(df.id.values))+1) #, theta=torch.exp
model.fit(df, ['week'], ['week'], df.id.values, 'weight', visualize=True)

# Non Linear Model
model = GMEM(1, 1, max(np.unique(df.id.values))+1,
             enc_in=[1, 4], enc_out=[4, 30, 1])  # theta=torch.exp
model.fit(df, ['week'], ['week'], df.id.values, 'weight',
          lr_f=0.005, visualize=True)