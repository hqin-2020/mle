import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore')
seed = 1
np.set_printoptions(suppress = True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

T = 300

λ = 0.5
η = 0

b11 = 1
b21 = 0.5 
b22 = 1 

As11 = 0.9
As12 = 0 
As13 = 0 
As14 = 0 
As21 = 0 
As22 = 0.8 
As23 = 0
As24 = 0 
As31 = 0 
As32 = 0 
As33 = 0.7 
As34 = 0 

Bs11 = 4
Bs21 = 0 
Bs22 = 3
Bs31 = 0 
Bs32 = 0 
Bs33 = 2

θ = np.array([λ, η, \
              b11, b21, b22, \
              As11, As12, As13, As14, \
              As21, As22, As23, As24, \
              As31, As32, As33, As34, \
              Bs11, Bs21, Bs22, Bs31, Bs32, Bs33])

Z0_true = np.array([[0.0],\
                    [0.0],\
                    [1.0]])
Az_true = np.array([[1, 1, 0],\
                    [0, λ, η],\
                    [0, 0, 1]])
Bz_true = np.array([[b11,  0],\
                    [b21, b22],\
                    [ 0,   0]])

S0_true = np.array([[0.0],\
                    [0.0],\
                    [0.0],\
                    [1.0]])
As_true = np.array([[As11, As12, As13, As14],\
                    [As21, As22, As23, As24],\
                    [As31, As32, As33, As34],\
                    [0.0,  0.0,  0.0,  1.0]])
Bs_true = np.array([[Bs11, 0.0,  0.0],\
                    [Bs21, Bs22, 0.0],\
                    [Bs31, Bs32, Bs33],\
                    [0.0,  0.0,  0.0]])

one3 = np.ones([3,1]) 

def sim_VAR(A, B, X0, shock_num, T):
    
    X_series = np.zeros([X0.shape[0], T+1])
    X_series[:, [0]]= X0
    
    W_series = np.random.multivariate_normal(np.zeros(shock_num), np.diag(np.ones(shock_num)), [T+1]).T
    
    for t in range(T):
        X_series[:,[t+1]] = A @ X_series[:,[t]] + B @ W_series[:,[t+1]]
        
    return X_series, W_series

def Kalman_Filter(obs, D, F, A, B, μ0, Σ0):
    
    state_μ = np.zeros([A.shape[1], obs.shape[1]])
    state_μ[:,[0]] = μ0
    state_Σ = np.zeros([A.shape[1], A.shape[1], obs.shape[1]])
    state_Σ[:,:,0] = Σ0
    
    ll = 0
    
    for t in range(obs.shape[1]-1):
        μt = state_μ[:,[t]]
        Σt = state_Σ[:,:,t]
        KΣt = (A@Σt@D.T + B@F.T)@np.linalg.inv(D@Σt@D.T+F@F.T)
        state_μ[:,[t+1]] = A@μt + KΣt@(obs[:,[t+1]] - D@μt)
        state_Σ[:,:,t+1] = A@Σt@A.T + B@B.T - (A@Σt@D.T + B@F.T)@np.linalg.inv(D@Σt@D.T + F@F.T)@(D@Σt@A.T+F@B.T)
        
        Ω = D@Σt@D.T + F@F.T
        ll += (-0.5*obs_series.shape[0]*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(Ω)) - 0.5*(obs[:,[t+1]] - D@μt).T@np.linalg.inv(Ω)@(obs[:,[t+1]] - D@μt))
        
    return state_μ, state_Σ ,ll

def ll(θ):
    
    λ, η, b11, b21, b22, As11, As12, As13, As14, As21, As22, As23, As24, As31, As32, As33, As34, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = θ
    
    A = np.array([[1,   1,   0,     0,     0,     0],\
                  [0,   λ,   0,     0,     0,     η],\
                  [0,   0,   As11,  As12,  As13,  As14],\
                  [0,   0,   As21,  As22,  As23,  As24],\
                  [0,   0,   As31,  As32,  As33,  As34],\
                  [0,   0,   0,     0,     0,     1]])
    B = np.array([[b11, 0,   0,     0,     0],\
                  [b21, b22, 0,     0,     0],\
                  [0,   0,   Bs11,  0,     0],\
                  [0,   0,   Bs21,  Bs22,  0],\
                  [0,   0,   Bs31,  Bs32,  Bs33],\
                  [0,   0,   0,       0,   0]])
    D = np.array([[1,   1,   As11,  As12,  As13,  As14],\
                  [1,   1,   As21,  As22,  As23,  As24],\
                  [1,   1,   As31,  As32,  As33,  As34]])
    F = np.array([[b11, 0,   Bs11,  0,     0],\
                  [b11, 0,   Bs21,  Bs22,  0],\
                  [b11, 0,   Bs31,  Bs32,  Bs33]])
    
    D0 = obs_series[:,[0]]
    
    try:
        μs = sp.linalg.solve(np.eye(3) - A[2:5,2:5], A[2:5,5:6])
        Σs = sp.linalg.solve_discrete_lyapunov(A[2:5,2:5], B[0:3,0:3])
        ones = np.ones([3,1])

        β = np.linalg.solve(np.hstack([Σs@np.array([[1,1],[0,-1],[-1,0]]), np.ones([3,1])]).T, np.array([[0,0,1]]).T)
        γ1 = np.array([[1],[0],[0]]) - (1/(ones.T@np.linalg.inv(Σs)@ones))*np.linalg.inv(Σs)@ones
        γ2 = np.array([[0],[1],[0]]) - (1/(ones.T@np.linalg.inv(Σs)@ones))*np.linalg.inv(Σs)@ones
        γ3 = np.array([[0],[0],[1]]) - (1/(ones.T@np.linalg.inv(Σs)@ones))*np.linalg.inv(Σs)@ones
        Γ = np.hstack([γ1, γ2, γ3])

        Z01 = β.T@(D0 - μs)
        Z02 = η/(1-λ)
        Σz01 = 0
        Σz02 = b22**2/(1-λ**2)
    
        S0 = Γ.T@(D0 - μs) + μs
        Σs0 = 1/(ones.T@np.linalg.inv(Σs)@ones)[0][0]
    
        μ0 = np.array([[Z01[0][0]],\
                       [Z02],\
                       [S0[0][0]],\
                       [S0[1][0]],\
                       [S0[2][0]],\
                       [1]])
        Σ0 = np.array([[Σz01,0,    0,   0,   0,   0],\
                       [0,   Σz02, 0,   0,   0,   0],\
                       [0,   0,    Σs0, Σs0, Σs0, 0],\
                       [0,   0,    Σs0, Σs0, Σs0, 0],\
                       [0,   0,    Σs0, Σs0, Σs0, 0],\
                       [0,   0,    0,   0,   0,   0]])
        _, _, ll = Kalman_Filter(obs_series, D, F, A, B, μ0, Σ0)
        
        return -ll[0][0]
    
    except:
        
        return np.nan

def callback(x):
    
    fobj = ll(x)
    θseries.append(x)
    llseries.append(fobj)


np.random.seed(seed)

optseries_sim = []
θseries_sim = []
llseries_sim = []

data_sim = 5
max_starting_point = 50

for i in tqdm(range(data_sim)):
    
    optseries = []
    θseries = []
    llseries = []
    Z_series, Wz = sim_VAR(Az_true, Bz_true, Z0_true, 2, T)
    S_series, Ws = sim_VAR(As_true, Bs_true, S0_true, 3, T)
    obs_series = one3@Z_series[[0],:] + S_series[0:3,:]
    
    count = 0
    while count < max_starting_point:
        try:
            start = np.concatenate((np.array((θ[0]+np.random.uniform(-0.2,0.2),θ[1]+np.random.uniform(0,0.2))),θ[2:]+np.random.uniform(-0.2,0.2,21)))
            bnds = ((0,1),(0,1),\
                    (-2,2),(-2,2),(-2,2),\
                    (-2,2),(-2,2),(-2,2),(-2,2),\
                    (-2,2),(-2,2),(-2,2),(-2,2),\
                    (-2,2),(-2,2),(-2,2),(-2,2),\
                    (-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5))
            count +=1
            θ_opt = sp.optimize.minimize(ll, start, method = 'L-BFGS-B', bounds = bnds, options = {'maxiter':10000, 'disp':False}, callback = callback)    
            optseries.append(θ_opt)
           
        except:
            optseries.append('Algebra Error')
            count +=1
    
    optseries_sim.append(optseries)
    θseries_sim.append(θseries)
    llseries_sim.append(llseries)

MLE1 = [optseries_sim, θseries_sim, llseries_sim]
with open('MLE1.pkl', 'wb') as f:
       pickle.dump(MLE1, f)

