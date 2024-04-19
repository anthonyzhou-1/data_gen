import h5py
from tqdm import tqdm

path1 = "./pde_data/2D/valid/2d_heat_adv_burgers_valid_downsampled.h5"
path2 = "./pde_data/2D/test/2d_heat_adv_burgers_test_downsampled.h5"

f1 = h5py.File(path1, 'r')
f2 = h5py.File(path2, 'r')

new_path = "./pde_data/2D/valid/2d_heat_adv_burgers_valid_downsampled_new.h5"
h5f = h5py.File(new_path, 'a')

split = 'valid'
dataset = h5f.create_group(split)

n_valid = f1[split]['u'].shape[0]
n_test = f2["test"]['u'].shape[0]
num_samples = n_valid + n_test
nt = f1[split]['u'].shape[1]
nx = f1[split]['u'].shape[2]
ny = f1[split]['u'].shape[3]

h5f_u = dataset.create_dataset(f'u', (num_samples, nt, nx, ny), dtype='f4')
coord = dataset.create_dataset(f'x', (2, nx, ny), dtype='f4')
tcoord = dataset.create_dataset(f't', (nt), dtype='f4')
h5f_ax = dataset.create_dataset(f'ax', (num_samples,), dtype='f4') 
h5f_ay = dataset.create_dataset(f'ay', (num_samples,), dtype='f4')
h5f_cx = dataset.create_dataset(f'cx', (num_samples,), dtype='f4')
h5f_cy = dataset.create_dataset(f'cy', (num_samples,), dtype='f4')
h5f_nu = dataset.create_dataset(f'nu', (num_samples,), dtype='f4')
#h5f_beta = dataset.create_dataset(f'beta', (num_samples), dtype='f4')
#h5f_gamma = dataset.create_dataset(f'gamma', (num_samples), dtype='f4')

coord[:] = f1[split]['x']
tcoord[:] = f1[split]['t']

for i in tqdm(range(n_valid)):
    h5f_u[i] = f1[split]['u'][i]
    h5f_ax[i] = f1[split]['ax'][i]
    h5f_ay[i] = f1[split]['ay'][i]
    h5f_cx[i] = f1[split]['cx'][i]
    h5f_cy[i] = f1[split]['cy'][i]
    h5f_nu[i] = f1[split]['nu'][i]

for i in tqdm(range(n_test)):
    h5f_u[i+n_valid] = f2["test"]['u'][i]
    h5f_ax[i+n_valid] = f2["test"]['ax'][i]
    h5f_ay[i+n_valid] = f2["test"]['ay'][i]
    h5f_cx[i+n_valid] = f2["test"]['cx'][i]
    h5f_cy[i+n_valid] = f2["test"]['cy'][i]
    h5f_nu[i+n_valid] = f2["test"]['nu'][i] 
    
#h5f_beta[:n_valid] = f1['valid']['beta']
#h5f_beta[n_valid:] = f2['test']['beta']
#h5f_gamma[:n_valid] = f1['valid']['gamma']
#h5f_gamma[n_valid:] = f2['test']['gamma']

f1.close()
f2.close()
h5f.close()