import numpy as np
import torch
from tqdm import tqdm

import h5py
from utils import RandomSin

###variable declarations
nx = 128
ny = 128

def simulate(nu, SAVE_STEPS, TOTAL_TIME, idx):
    # Define constants
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.0001*nu/(dx*dy)
    dt = sigma * dx * dy / nu

    nt = int(TOTAL_TIME/dt)
    if(nu*dt/(dx**2) >= 0.5):
        raise ValueError("Unstable Simulation.")
    SAVE_EVERY = int(nt/SAVE_STEPS) + 1

    # Define domain and solutions
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    grid = np.meshgrid(x, y)
    u = torch.zeros((ny, nx))  
    un = np.zeros((ny, nx))

    ###Assign initial conditions
    f = RandomSin((nx, ny))
    u = f.sample(grid=grid, seed=idx)

    all_us = torch.empty((SAVE_STEPS, nx, ny))
    times = torch.empty(SAVE_STEPS)
    all_us[0] = u.clone()
    
    for n in range(nt-1): ##loop across number of time steps
        un = u.clone()
    
        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy
    
        # Calculate update
        u = nu*dt*diff_u/dx**2 + u
    
        if((n+1)%SAVE_EVERY == 0):
            all_us[(n)//SAVE_EVERY] = u
            times[(n)//SAVE_EVERY] = TOTAL_TIME*(n+1)/nt

    return all_us, grid, times
            

def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 10
    nt = 10000
    inits = 200
    nus = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    for j in [-5,-4]:
        for i in np.arange(1, 10, 1):
            nus.append(float("{0}e{1}".format(i, j)))
    nus.append(1e-3)
    h5f = h5py.File("../2D_NS_DATA/gaussian_2d_Heat_{}s_{}inits_{}nus_ns_match.h5".format(TOTAL_TIME, inits,
                                                                       len(nus)), 'w')
    idx = 0
    for nu in nus:
        print("NU: {0:.4f}, TIMESTEPS: {1}".format(nu, nt))
        for i in tqdm(range(inits)):
            key = 'Heat_{}_{}'.format(nu, i)
            u, grid, times = simulate(nu, SAVE_STEPS, TOTAL_TIME, idx)

            dataset = h5f.create_group(key)
            dataset['u'] = u
            dataset['nu'] = nu
            dataset['grid'] = grid
            dataset['time'] = times
            idx += 1


if __name__ == '__main__':
    main()
