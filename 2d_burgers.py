import numpy as np
import torch
from utils import RandomSin
from scipy.stats import uniform
import h5py
from tqdm import tqdm

###variable declarations
J = 5
lj = 3
length = 2

def simulate(nu, cx, cy, nx, ny, SAVE_STEPS, TOTAL_TIME, idx):
    nx = 2*nx # Double the resolution for stability
    ny = 2*ny 

    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    dt = 5e-5
    nt = int(TOTAL_TIME/dt)
    SAVE_EVERY = nt//SAVE_STEPS

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    grid = np.meshgrid(x, y)

    u = torch.zeros((ny, nx))  
    v = torch.zeros((ny, nx))
    un = np.zeros((ny, nx))
    vn = np.zeros((ny, nx))

    ###Assign initial conditions
    f = RandomSin((nx, ny), J, lj, length)
    u = torch.Tensor(f.sample(grid=grid, seed=idx))
    v = torch.Tensor(f.sample(grid=grid, seed=idx))
    u0 = u.clone()
    v0 = v.clone()

    #for n in tqdm(range(nt + 1)): ##loop across number of time steps
    all_us = torch.empty((SAVE_STEPS, nx, ny))
    all_vs = torch.empty((SAVE_STEPS, nx, ny))
    times = torch.empty(SAVE_STEPS)

    ## Save initial condition
    all_us[0] = u0
    all_vs[0] = v0
    times[0] = 0

    for n in tqdm(range(nt-1)): ##loop across number of time steps -1 because we already have the initial condition

        # Calculate finite differences
        un = u.clone()
        vn = v.clone()

        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy

        diff_vx = (torch.roll(vn, shifts=(1), dims=(1)) + torch.roll(vn, shifts=(-1), dims=(1)) - 2*vn)
        diff_vy = (torch.roll(vn, shifts=(1), dims=(0)) + torch.roll(vn, shifts=(-1), dims=(0)) - 2*vn)
        diff_v = diff_vx + diff_vy

        # Calculate finite differences for nonlinear advection term
        if(cx <= 0 and cy >= 0):
            adv_u = -cx*un*(un - torch.roll(un, shifts=(-1), dims=(1))) + cy*vn*(un - torch.roll(un, shifts=(1), dims=(0)))
            adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy >= 0):
            adv_u = cx*un*(un - torch.roll(un, shifts=(1), dims=(1))) + cy*vn*(un - torch.roll(un, shifts=(1), dims=(0)))
            adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))
        elif(cx <= 0 and cy <= 0):
            adv_u = -cx*un*(un - torch.roll(un, shifts=(-1), dims=(1))) - cy*vn*(un - torch.roll(un, shifts=(-1), dims=(0)))
            adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy <= 0):
            adv_u = cx*un*(un - torch.roll(un, shifts=(1), dims=(1))) - cy*vn*(un - torch.roll(un, shifts=(-1), dims=(0)))
            adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))

        # Calculate update
        u = nu*dt*diff_u/dx**2 - dt*adv_u/dx + u
        v = nu*dt*diff_v/dy**2 - dt*adv_v/dy + v

        if torch.isnan(u).any():
            raise ValueError('UNSTABLE')
    

        if((n+1)%SAVE_EVERY == 0):
            all_us[(n+1)//SAVE_EVERY] = u.clone()
            all_vs[(n+1)//SAVE_EVERY] = v.clone()
            times[(n+1)//SAVE_EVERY] = TOTAL_TIME*(n+1)/nt


    return all_us, all_vs, grid, times

            
def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 2
    nx = 256
    ny = 256

    num_samples = 1024
    nu_low = 7.5e-3
    nu_high = 1.5e-2
    nus = uniform.rvs(nu_low, nu_high-nu_low, size=num_samples) # Sample from uniform distribution [7.5e-5, 1.5e-2]

    c_high = 1.0
    c_low = 0.5
    cxs = uniform.rvs(c_low, c_high-c_low, size=num_samples) 
    cys = uniform.rvs(c_low, c_high-c_low, size=num_samples) 
    h5f = h5py.File(f"./pde_data/2d_Burgers_{num_samples}ns_{nx}nx_{ny}ny_{nu_low}_{nu_high}nu_{c_low}_{c_high}c.h5", 'w')

    for i in tqdm(range(num_samples)):
        nu = nus[i]
        cx = cxs[i]
        cy = cys[i]
        
        print("NU: {0:.4f}, CX: {1:.4f}, CY: {2:.4f}".format(nu, cx, cy))
        key = 'Burgers_{0:.4f}_{1:.4f}_{2:.4f}'.format(nu, cx, cy)
        u, v, grid, times = simulate(nu, cx, cy, nx, ny, SAVE_STEPS, TOTAL_TIME, i + 5000)

        u = u[:, ::2, ::2]
        v = v[:, ::2, ::2]
        grid = (grid[0][::2, ::2], grid[1][::2, ::2])

        dataset = h5f.create_group(key)
        dataset['u'] = u
        dataset['v'] = v
        dataset['nu'] = nu
        dataset['c'] = [cx, cy]
        dataset['grid'] = grid
        dataset['time'] = times


if __name__ == '__main__':
    main()
