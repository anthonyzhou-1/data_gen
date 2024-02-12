import numpy as np
import torch
from torch.nn import functional as F
from typing import Tuple
from mp_code.PDEs import *
import matplotlib.pyplot as plt
from FyeldGenerator import generate_field
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors

def vis_1D(traj, ax=None, title=""):
    '''
    Plots 1D PDE trajectory, with lighter colors for later time steps
    Traj is expected in shape: [nt, nx]
    '''
    N = traj.shape[0]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))
    x = torch.linspace(0, 2, traj.shape[1])
    for i in range(N):
        ax.plot(x, traj[i])
    ax.set_title(title)

def vis_1D_im(u, ax = None, fig=None, title='test', aspect='auto'):
    '''
    Plots 1D PDE trajectory as an image
    Traj is expected in shape: [nt, nx]
    '''
    im = ax.imshow(u, aspect=aspect)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

def grf(shape:Tuple, alpha:float):
    '''
    Generates a Gaussian Random Field
    '''
    # Helper that generates power-law power spectrum
    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)
        return Pk

    # Draw samples from a normal distribution
    def distrib(shape):
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    field = generate_field(distrib, Pkgen(alpha), shape)
    return field

class RandomSin():

    def __init__(self, 
                 shape:Tuple,
                 num_waves:int =5, 
                 max_wavenum:int = 3, 
                 length:float = 2.0):
        '''
        shape: tuple, shape of the grid
        num_waves: int, number of waves to sum
        max_wavenum: int, maximum wavenumber
        length: float, length of the grid
        '''
        self.shape = shape
        self.num_waves = num_waves
        self.max_wavenum = max_wavenum
        self.length = length

    def sample(self, grid = None, seed = None):
        # Samples random summation of sine waves according to MP-PDE Solvers (and also loosely PDEBench)

        # Set seed if given
        if seed is not None:
            np.random.seed(seed)

        # Setting constants 
        J = self.num_waves
        lj = self.max_wavenum + 1
        nx = self.shape[0]
        ny = self.shape[1]
        L = self.length

        # Generate grid if not given
        if grid is None:
            x = np.linspace(-1, 1, nx)
            y = np.linspace(-1, 1, ny)
            [xx, yy] = np.meshgrid(x, y)   
        else:
            xx = grid[0]
            yy = grid[1]

        u = np.zeros((nx, ny))

        A = np.random.uniform(-0.5, 0.5, J)
        Kx = 2*np.pi*np.random.randint(1, lj, J)/L
        Ky = 2*np.pi*np.random.randint(1, lj, J)/L
        phi = np.random.uniform(0, 2*np.pi, J)

        for i in range(J):
            u = u + A[i]*np.sin(Kx[i] * xx + Ky[i] * yy + phi[i])
        return u

def vis_2d(u, title="", cmap = "seismic"):
    '''
    Makes gif of 2D PDE trajectory
    u is expected in shape [nt, nx, ny]
    '''

    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    ax.set_title(title)
    for i in range(u.shape[0]):
        im = ax.imshow(u[i].squeeze(), animated=True, cmap=cmap)
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save(f"assets/{title}.gif", writer=writer)
    print("saved")

def vis_2d_plot(u, grid, title="", downsample=1):
    '''
    Plots 2D PDE trajectory on a 3D plot as an mp4
    u is expected in shape [nt, nx, ny]
    grid is expected in shape [nx, ny]
    '''

    norm = colors.Normalize()
    X, Y = grid
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    if downsample > 1:
        X = X[::downsample, ::downsample]
        Y = Y[::downsample, ::downsample]
        u = u[::downsample, ::downsample, ::downsample]
    nt, nx, ny = u.shape

    def animate(n):
        ax.cla()

        x = u[n]
        cmap = cm.jet(norm(x))
        ax.plot_surface(X, Y, x, facecolors=cmap, rstride=1, cstride=1)
        ax.set_zlim(-1.5, 1.5)

        return fig,


    anim = FuncAnimation(fig = fig, func = animate, frames = nt, interval = 1, repeat = False)
    anim.save(f'assets/{title}.mp4',fps=10)