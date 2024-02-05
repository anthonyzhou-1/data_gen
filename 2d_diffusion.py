import numpy
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import scipy
from sklearn.datasets import make_spd_matrix

import h5py
import scipy.io
from io import StringIO
import tokenize
from matplotlib import pyplot as plt
from sympy import Function, pprint, Sum, sin, cos, log
from sympy.core.symbol import Symbol, symbols
from sympy.abc import x, t, y, alpha, beta, gamma, delta, A, omega, pi, l, L, phi, j, J, eta, nu,     w#, nabla
from sympy.vector import gradient, CoordSys3D, Del, divergence
from sympy.physics.vector import dot, ReferenceFrame

#torch.set_default_device('cuda:0')

WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'v', 'y',
         'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',     '10^', 'E', 'e', ',', '.', '&']
word2id = {w: i for i, w in enumerate(WORDS)}
id2word = {i: w for i, w in enumerate(WORDS)}

# Random seeds
numpy.random.seed(137)
torch.manual_seed(137)

###variable declarations
nx = 128
ny = 128
#nt = 1000
#cx, cy = -2, 1
#cx, cy = 1, 2
#cx, cy = -2, -1


def get_tokens(nu):
    u = Function('u')
    t = Symbol('t')
    u = u(t,x,y)
    ux = u.diff(x)
    uxx = -ux.diff(x)
    uy = u.diff(y)
    uyy = -uy.diff(y)
    ut = u.diff(t)
    utt = ut.diff(t)


    right = nu * uxx + nu * uyy

    left_str = StringIO(str(ut)).readline
    left_tokens = []
    for idx, t in enumerate(tokenize.generate_tokens(left_str)):
        if(t.type not in [0, 4]):
            left_tokens.append(t.string)
            #print(idx, t.string)
    #print(left_tokens)


    #pprint(right)
    right_str = StringIO(str(right)).readline
    right_tokens = []
    for idx, t in enumerate(tokenize.generate_tokens(right_str)):
        if(t.type not in [0, 4]):
            #if(idx == 6):
            #    #right_tokens.append("dot")
            #    right_tokens.append("Delta")
            #else:
            right_tokens.append(t.string)
            #print(idx, t.string)

    #alt_right_str = StringIO(str(u)).readline
    #alt_right_tokens = ["-"]
    #alt_right_tokens.extend(list(str(velocity**2)[:4]))
    #alt_right_tokens.append("*")
    #alt_right_tokens.append("Delta")

    #for idx, t in enumerate(tokenize.generate_tokens(alt_right_str)):
    #    if(t.type not in [0,4]):
    #        alt_right_tokens.append(t.string)

    # Do we even need the initial condition?
    # This leaves the forcing term and the viscosity as things to modify.
    all_tokens = []
    all_tokens.extend(left_tokens)
    all_tokens.append("=")
    all_tokens.extend(right_tokens)

    return all_tokens


def encode_tokens(all_tokens):
    encoded_tokens = []
    num_concat = 0
    for i in range(len(all_tokens)):
        try: # All the operators, bcs, regular symbols
            encoded_tokens.append(word2id[all_tokens[i]])
            if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                num_concat += 1
        except KeyError: # Numerical values
            if(isinstance(all_tokens[i], str)):
                for v in all_tokens[i]:
                    encoded_tokens.append(word2id[v])
                if(num_concat >= 5): # We're in a list of sampled parameters
                    encoded_tokens.append(word2id[","])
            else:
                raise KeyError("Unrecognized token: {}".format(all_tokens[i]))

    return encoded_tokens


def simulate(nu, SAVE_STEPS, TOTAL_TIME, idx):

    np.random.seed(idx)
    torch.manual_seed(idx)

    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.0001*nu/(dx*dy)
    dt = sigma * dx * dy / nu
    #print("dt: {}".format(dt))
    nt = int(TOTAL_TIME/dt)
    #nt = 10000
    #print("nt: {}".format(nt))
    #print("STABLE: {0}, {1:.5f}".format(nu*dt/(dx**2) < 0.5, nu*dt/(dx*dy)))
    if(nu*dt/(dx**2) >= 0.5):
        raise ValueError("Unstable Simulation.")
    SAVE_EVERY = int(nt/SAVE_STEPS) + 1
    #print("SAVE EVERY: {}".format(SAVE_EVERY))
    #nt = 6000
    #raise
    #print("STABLE: {0}, {1:.5f}".format(2*nu*dt/(dx**2) < 0.5, 2*nu*dt/(dx*dy)))
    #print("SAVE EVERY {} STEPS".format(nt/SAVE_STEPS))
    #raise


    ###########################################
    x = numpy.linspace(-1, 1, nx)
    y = numpy.linspace(-1, 1, ny)

    u = torch.zeros((ny, nx))  # create a 1xn vector of 1's
    v = torch.zeros((ny, nx))
    un = numpy.zeros((ny, nx))
    vn = numpy.zeros((ny, nx))
    comb = numpy.zeros((ny, nx))

    ###Assign initial conditions

    ###set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    #u[int(.4 / dy):int(0.6 / dy + 1),int(.4 / dx):int(0.6 / dx + 1)] = 1
    ###set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    #v[int(.4 / dy):int(0.6 / dy + 1),int(.4 / dx):int(0.6 / dx + 1)] = 1
    x_start = numpy.random.choice(numpy.arange(48, 80, 1))
    y_start = numpy.random.choice(numpy.arange(48, 80, 1))
    #x_end = numpy.random.choice(numpy.arange(0.1, 0.91, 0.1)[
    #                            numpy.arange(0.1, 0.91, 0.1) > x_start])
    #y_end = numpy.random.choice(numpy.arange(0.1, 0.91, 0.1)[
    #                            numpy.arange(0.1, 0.91, 0.1) > y_start])
    #print(y_start/dy, y_end/dy, x_start/dy, x_end/dy)
    #print(x_start, y_start)

    sigma = 0.1
    distribution = lambda x, x0: 1/(2.*np.pi * sigma**2) * np.exp(-(np.linalg.norm(x-x0, axis=2)**    2)/(2.*sigma**2))

    mat = make_spd_matrix(2, random_state=idx)
    #print(mat)
    eig = np.linalg.eig(mat)[0][0]
    mat = mat/(20*eig)
    #print(np.linalg.eig(mat)[0])

    X, Y = numpy.meshgrid(x, y)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    grid = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1)
    #print(x_start, y_start)
    #print(grid.shape)
    #print(grid[x_start][y_start])
    u = scipy.stats.multivariate_normal.pdf(grid, mean=grid[x_start][y_start], cov=mat)
    #print(u.shape)
    u = torch.Tensor(u)
    #print(u.max())
    #print(u.min())
    u0 = u.clone()
    ###########################################

    
    ###(plot ICs)
    #fig = pyplot.figure(figsize=(11, 7), dpi=100)
    #ax = fig.add_subplot(projection='3d')
    #X, Y = numpy.meshgrid(x, y)
    #ax.plot_surface(X, Y, u[:], cmap=cm.viridis, rstride=1, cstride=1)
    #ax.plot_surface(X, Y, v[:], cmap=cm.viridis, rstride=1, cstride=1)
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    
    #for n in tqdm(range(nt + 1)): ##loop across number of time steps
    all_us = torch.empty((100, 128, 128))
    times = torch.empty(100)
    for n in range(nt): ##loop across number of time steps
        un = u.clone()
    
        # Calculate finite differences
        un = u.clone()
    
        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy
    
        # Calculate update
        u = nu*dt*diff_u/dx**2 + u
    
        if((n+1)%SAVE_EVERY == 0):
            #print(n+1, (n)//100, TOTAL_TIME*(n+1)/nt)
            all_us[(n)//SAVE_EVERY] = u
            times[(n)//SAVE_EVERY] = TOTAL_TIME*(n+1)/nt

    #print(times)
    #raise
    #fig = pyplot.figure(figsize=(11, 7), dpi=100)
    #ax = fig.add_subplot(projection='3d')
    #X, Y = numpy.meshgrid(x, y)
    ##ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=1, cstride=1)
    #ax.plot_surface(X, Y, u0, cmap=cm.Reds, rstride=1, cstride=1)
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    #plt.show()
    #raise

    #for i in range(0, 100):
    #    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    #    ax = fig.add_subplot(projection='3d')
    #    X, Y = numpy.meshgrid(x, y)
    #    ax.plot_surface(X, Y, all_us[i], cmap=cm.viridis, rstride=1, cstride=1)
    #    ax.set_xlabel('$x$')
    #    ax.set_ylabel('$y$')
    #    plt.savefig("./burgers_example.png")
    #    plt.show()
    #raise


    X = torch.tensor(X)
    Y = torch.tensor(Y)
    #print(all_us)
    #all_us = torch.tensor(all_us)
    return u0, all_us, torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1), times
            

def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 10
    nt = 10000
    inits = 200
    nus = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    nus = []
    #for j in [-9, -8, -7, -6, -5]:
    #for j in [-8, -7, -6]:
    #for j in [-4, -3, -2]:
    for j in [-5,-4]:
        for i in numpy.arange(1, 10, 1):
            nus.append(float("{0}e{1}".format(i, j)))
    nus.append(1e-3)
    h5f = h5py.File("../2D_NS_DATA/gaussian_2d_Heat_{}s_{}inits_{}nus_ns_match.h5".format(TOTAL_TIME, inits,
                                                                       len(nus)), 'w')
    #h5f = h5py.File("../2D_NS_DATA/tempplot_Heat.h5".format(TOTAL_TIME, inits,
    #                                                                   len(nus)), 'w')

    idx = 0
    for nu in nus:
        print("NU: {0:.4f}, TIMESTEPS: {1}".format(nu, nt))
        for i in tqdm(range(inits)):
            key = 'Heat_{}_{}'.format(nu, i)
            u0, u, grid, times = simulate(nu, SAVE_STEPS, TOTAL_TIME, idx)
            #print(grid.shape)
            #print(u0.shape)
            #print(u.shape)

            raw_tokens = get_tokens(nu)
            tokens = encode_tokens(raw_tokens)

            #print(raw_tokens)
            #print(tokens)

            dataset = h5f.create_group(key)
            dataset['tokens'] = tokens
            dataset['u0'] = u0
            dataset['u'] = u
            dataset['nu'] = nu
            dataset['grid'] = grid
            dataset['time'] = times
            idx += 1
            #raise


if __name__ == '__main__':
    main()
