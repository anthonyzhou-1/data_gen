import h5py
from tqdm import tqdm
import numpy as np

path = "/home/cmu/anthony/data_gen/pde_data/2D/train/2d_heat_adv_burgers.h5"
f = h5py.File(path, 'r')

h5f = h5py.File(f"./pde_data/2D/train/2d_heat_adv_burgers_torch.h5", 'w')

grid = f_heat[heat_keys[0]]["grid"]
grid = np.array(grid)[:, ::4, ::4] # downsample to 64x64
time = f_heat[heat_keys[0]]["time"]

h5f.create_dataset("grid", data=grid)
h5f.create_dataset("time", data=time)

num_samples = 1024

for i in tqdm(range(num_samples)):

    key = heat_keys[i]
    dataset = h5f.create_group(key)
    dataset['u'] = np.array(f_heat[key]['u'])[:, ::4, ::4]
    coeffs = np.zeros(5)
    nu = np.array(f_heat[key]['nu']).item()
    coeffs[0] = nu
    dataset['coeffs'] = coeffs
    dataset['pde'] = np.array(['heat'], dtype='S')

    key = adv_keys[i]
    dataset = h5f.create_group(key)
    dataset['u'] = np.array(f_adv[key]['u'])[:, ::4, ::4]
    coeffs = np.zeros(5)
    cs = np.array(f_adv[key]['coeffs'])
    coeffs[1] = cs[0]
    coeffs[2] = cs[1]
    dataset['coeffs'] = coeffs
    dataset['pde'] = np.array(['advection'], dtype='S')

    key = burgers_keys[i]
    dataset = h5f.create_group(key)
    dataset['u'] = np.array(f_burgers[key]['u'])[:, ::4, ::4]
    coeffs = np.zeros(5)
    nu = np.array(f_burgers[key]['nu']).item()
    coeffs[0] = nu
    cs = np.array(f_burgers[key]['c'])
    coeffs[3] = cs[0]
    coeffs[4] = cs[1]
    dataset['coeffs'] = coeffs
    dataset['pde'] = np.array(['burgers'], dtype='S')

h5f.close()
