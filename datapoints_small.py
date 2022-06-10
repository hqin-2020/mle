import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore')
batch_n = np.random.randint(1,100000)
from concurrent.futures import ProcessPoolExecutor
np.set_printoptions(suppress = True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from optimize import *
import time 
import os

time_start = time.time()
data_points = 2
start_trials = 14*3

T = 300
λ, η = 0.5, 0
b11, b21 = 1, 0.5 

As11, As12, As13, As14 = 0.9, 0.0, 0.0, 0
As21, As22, As23, As24 = 0.0, 0.8, 0.0, 0
As31, As32, As33, As34 = 0.0, 0.0, 0.7, 0

Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = 4, 0, 3, 0, 0, 2

Az_true = np.array([[1, 1, 0],\
                    [0, λ, η],\
                    [0, 0, 1]])
Bz_true = np.array([[b11],\
                    [b21],\
                    [0]])
As_true = np.array([[As11, As12, As13, As14],\
                    [As21, As22, As23, As24],\
                    [As31, As32, As33, As34],\
                    [0.0,  0.0,  0.0,  1.0]])
Bs_true = np.array([[Bs11, 0.0,  0.0],\
                    [Bs21, Bs22, 0.0],\
                    [Bs31, Bs32, Bs33],\
                    [0.0,  0.0,  0.0]])

ones = np.ones([3,1])

θ = np.array([λ, η, \
              b11, b21, \
              As11, As12, As13, As14, \
              As21, As22, As23, As24, \
              As31, As32, As33, As34, \
              Bs11, Bs21, Bs22, Bs31, Bs32, Bs33])

def sim_VAR(A, B, X0, shock_num, T):

    X_series = np.zeros([X0.shape[0], T+1])
    X_series[:, [0]]= X0

    W_series = np.random.multivariate_normal(np.zeros(shock_num), np.diag(np.ones(shock_num)), [T+1]).T

    for t in range(T):
        X_series[:,[t+1]] = A @ X_series[:,[t]] + B @ W_series[:,[t+1]]

    return X_series, W_series

def sim_obs(As_true, Bs_true, η, λ, b21):

    μs = sp.linalg.solve(np.eye(3) - As_true[0:3,0:3], As_true[0:3,3:4])
    Σs = sp.linalg.solve_discrete_lyapunov(As_true[0:3,0:3], Bs_true[0:3,0:3]@Bs_true[0:3,0:3].T)

    μz01 = 0
    Σz01 = 100
    Z01 = np.random.normal(μz01, Σz01)

    μz02 = η/(1-λ)
    Σz02 = b21**2/(1-λ**2)
    Z02 = np.random.normal(μz02, Σz02)

    Z0_true = np.array([[Z01],[Z02],[1]])
    S0_true = np.append(np.random.multivariate_normal(μs.flatten(), Σs),np.array(1)).reshape([-1,1])

    Z_series, Wz = sim_VAR(Az_true, Bz_true, Z0_true, 1, T)
    S_series, Ws = sim_VAR(As_true, Bs_true, S0_true, 3, T)
    obs_series = ones@Z_series[[0],:] + S_series[0:3,:]
    
    return obs_series

obs_list = []
for i in range(data_points):
    obs = sim_obs(As_true, Bs_true, η, λ, b21)
    obs_series = [obs for _ in range(start_trials)] 
    obs_list.append(obs_series)

batch = []
if __name__ == '__main__':
    for i in tqdm(range(data_points)):
        with ProcessPoolExecutor() as pool:
            results = pool.map(optimization, obs_list[i])
        results = [r for r in results]
        batch.append(results)

with open('MLE_'+str(batch_n)+'.pkl', 'wb') as f:
       pickle.dump(batch, f)

time_end = time.time()
time_elapsed = time_end - time_start
print(time_elapsed)