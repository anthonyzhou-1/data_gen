import h5py
from tqdm import tqdm
split = 'valid'
path = "./pde_data/2D/valid/Heat_Adv_Burgers_768_OOD.h5"
#path = f"./pde_data/2D/{split}/2d_heat_adv_burgers_{split}.h5"

f = h5py.File(path, 'r')

new_path = f"./pde_data/2D/{split}/2d_heat_adv_burgers_{split}_downsampled_OOD.h5"
h5f = h5py.File(new_path, 'a')

dataset = h5f.create_group(split)

num_samples = len(f[split]['u'])
nx = 32
ny = 32
nt = 32

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

coord[:] = f[split]['x'][:, ::2, ::2]
tcoord[:] = f[split]['t'][:nt]

for i in tqdm(range(num_samples)):
    h5f_u[i] = f[split]['u'][i][:nt, ::2, ::2]
    h5f_ax[i] = f[split]['ax'][i]
    h5f_ay[i] = f[split]['ay'][i]
    h5f_cx[i] = f[split]['cx'][i]
    h5f_cy[i] = f[split]['cy'][i]
    h5f_nu[i] = f[split]['nu'][i]

f.close()
h5f.close()