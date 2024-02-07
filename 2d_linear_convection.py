import numpy as np
import torch
from sklearn.datasets import make_spd_matrix
from utils import RandomSin
import h5py


def simulate(cx, cy, nx, ny, SAVE_STEPS, TOTAL_TIME, seed):
    dt = TOTAL_TIME/SAVE_STEPS
    nt = int(np.ceil(TOTAL_TIME/dt))

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)

    # Generate initial Gaussian pulse
    grid = np.meshgrid(x, y)
    f = RandomSin((nx, ny))
    u = torch.Tensor(f.sample(grid=grid, seed=seed))

    all_us = torch.empty((nt, nx, ny))
    all_vs = torch.empty((nt, nx, ny))
    times = torch.empty(nt)

    for n in range(1, nt+1): ##loop across number of time steps
        x_adv = -dt*n*cx
        y_adv = -dt*n*cy

        # Make new grid and subtract c*t
        new_x = x - x_adv
        new_y = y - y_adv
        new_grid = np.meshgrid(new_x, new_y)

        # Sample function at new grid
        new_u = torch.Tensor(f.sample(grid=new_grid, seed=seed))
        
        all_us[n-1] = new_u.clone()
        times[n-1] = TOTAL_TIME*(n)/nt

    return all_us, grid, times

def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 10
    num_samples = 1024
    nx = 128
    ny = 128
    h5f = h5py.File(f"./pde_data/2d_adv_{num_samples}ns_{nx}nx_{ny}ny", 'w')

    for i in range(num_samples):
        cx = 5 * torch.rand(1).item()
        cy = 5 * torch.rand(1).item()
        print("CX: {0:.4f}, CY: {1:.4f}".format(cx, cy))
        key = '{0:.4f}_{1:.4f}'.format(cx, cy)
        u, grid, times = simulate(cx, cy, nx, ny, SAVE_STEPS, TOTAL_TIME, i)

        dataset = h5f.create_group(key)
        dataset['coeffs'] = [cx, cy]
        dataset['u'] = u
        dataset['grid'] = grid
        dataset['time'] = times

if __name__ == '__main__':
    main()

