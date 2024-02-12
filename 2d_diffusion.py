import numpy as np
import torch

import h5py
from utils import RandomSin
from scipy.stats import uniform

###variable declarations
nx = 256
ny = 256
J = 5
lj = 3
length = 2

def simulate(nu, nx, ny, SAVE_STEPS, TOTAL_TIME, idx):
    dx = 2 / (nx - 1)
    dt = 0.0001
    nt = int(TOTAL_TIME/dt)
    if(nu*dt/(dx**2) >= 0.5):
        raise ValueError("Unstable Simulation.")
    SAVE_EVERY = int(nt/SAVE_STEPS)

    # Define domain and solutions
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    grid = np.meshgrid(x, y)
    u = torch.zeros((ny, nx))  
    un = np.zeros((ny, nx))

    ###Assign initial conditions
    f = RandomSin((nx, ny))
    u = torch.Tensor(f.sample(grid=grid, seed=idx))
    u0 = u.clone()

    all_us = torch.empty((SAVE_STEPS, nx, ny))
    times = torch.empty(SAVE_STEPS)

    ## Save initial condition
    all_us[0] = u0
    times[0] = 0

    for n in range(nt-1): ##loop across number of time steps (nt-1 because we already have the initial condition)
        un = u.clone()

        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy

        # Calculate update
        u = nu*dt*diff_u/dx**2 + u

        if((n+1)%SAVE_EVERY == 0):
            all_us[(n+1)//SAVE_EVERY] = u
            times[(n+1)//SAVE_EVERY] = TOTAL_TIME*(n+1)/nt

    return all_us, grid, times
            

def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 2
    num_samples = 1024
    high = 2e-2
    low = 3e-3
    h5f = h5py.File(f"./pde_data/2d_heat_{num_samples}ns_{nx}nx_{ny}ny_{low}_{high}_nu.h5", 'w')

    nus = uniform.rvs(low, high-low, size=num_samples) # Sample from uniform distribution [3e-3, 2e-2]
    for i in range(num_samples):
        nu = nus[i]
        key = 'Heat_{0:.8f}'.format(nu)
        print("NU: {0:.8f}".format(nu))
        u, grid, times = simulate(nu, SAVE_STEPS, TOTAL_TIME, i)

        dataset = h5f.create_group(key)
        dataset['u'] = u
        dataset['nu'] = nu
        dataset['grid'] = grid
        dataset['time'] = times

if __name__ == '__main__':
    main()
