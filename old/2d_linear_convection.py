import numpy as np
import torch
from utils import RandomSin
import h5py
from scipy.stats import uniform
import argparse

nx = 256
ny = 256
J = 5
lj = 3
length = 2

def simulate(cx, cy, SAVE_STEPS, TOTAL_TIME, seed):
    dt = TOTAL_TIME/SAVE_STEPS
    nt = int(np.ceil(TOTAL_TIME/dt))

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)

    # Generate initial condition
    grid = np.meshgrid(x, y)
    f = RandomSin((nx, ny), J, lj, length)
    u = torch.Tensor(f.sample(grid=grid, seed=seed))

    all_us = torch.empty((nt, nx, ny))
    times = torch.empty(nt)
    all_us[0] = u.clone()
    times[0] = 0

    for n in range(1, nt): ##loop across number of time steps -1 because we already have the initial condition
        x_adv = -dt*n*cx
        y_adv = -dt*n*cy

        # Make new grid and subtract c*t
        new_x = x - x_adv
        new_y = y - y_adv
        new_grid = np.meshgrid(new_x, new_y)

        # Sample function at new grid
        new_u = torch.Tensor(f.sample(grid=new_grid, seed=seed))
        
        all_us[n] = new_u.clone()
        times[n] = TOTAL_TIME*(n)/nt

    return all_us, grid, times

def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 2
    num_samples = 128
    low = 0.1
    high = 2.5
    h5f = h5py.File(f"./pde_data/2d_adv_test_{num_samples}ns_{nx}nx_{ny}ny_{low}_{high}_c.h5", 'w')

    cxs = uniform.rvs(low, high-low, size=num_samples) # Sample from uniform distribution [.1, 2.5]
    cys = uniform.rvs(low, high-low, size=num_samples) # Sample from uniform distribution [.1, 2.5]

    for i in range(num_samples):
        cx = cxs[i]
        cy = cys[i]
        print("CX: {0:.4f}, CY: {1:.4f}".format(cx, cy))
        key = '{0:.4f}_{1:.4f}'.format(cx, cy)
        u, grid, times = simulate(cx, cy, SAVE_STEPS, TOTAL_TIME, i + 20000)

        dataset = h5f.create_group(key)
        dataset['coeffs'] = [cx, cy]
        dataset['u'] = u
        dataset['grid'] = grid
        dataset['time'] = times

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--experiment', type=str, default='KdV',
                        help='Experiment for which data should create for: [KdV, KS, Burgers]')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--nt', type=int, default=250,
                        help='Time steps used for solving')
    parser.add_argument('--nx', type=int, default=256,
                        help='Spatial resolution')
    parser.add_argument('--L', type=float, default=128.,
                        help='Length for which we want to solve the PDE')
    parser.add_argument('--train_samples', type=int, default=2 ** 5,
                        help='Samples in the training dataset')
    parser.add_argument('--valid_samples', type=int, default=2 ** 5,
                        help='Samples in the validation dataset')
    parser.add_argument('--test_samples', type=int, default=2 ** 5,
                        help='Samples in the test dataset')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size used for creating training, val, and test dataset. So far the code only works for batch_size==1')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for additional datasets')
    parser.add_argument('--log', type=eval, default=False,
                        help='pip the output to log file')

    args = parser.parse_args()
    main(args)

