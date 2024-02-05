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
from sympy.abc import x, t, y, alpha, beta, gamma, delta, A, omega, pi, l, L, phi, j, J, eta, nu,     w, v
from sympy.vector import gradient, CoordSys3D, Del, divergence
from sympy.physics.vector import dot, ReferenceFrame

WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'v', 'y',
         'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',     '10^', 'E', 'e', ',', '.', '&']
word2id = {w: i for i, w in enumerate(WORDS)}
id2word = {i: w for i, w in enumerate(WORDS)}

# Random seeds
numpy.random.seed(137)
torch.manual_seed(137)

###variable declarations
#nx = 64
#ny = 64
nx = 128
ny = 128

#nt = 1000
#cx, cy = -2, 1
#cx, cy = 1, 2
#cx, cy = -2, -1


def get_tokens(nu, cx, cy):
    u = Function('u')
    v = Function('v')
    t = Symbol('t')
    u = u(t,x,y)
    ux = u.diff(x)
    uxx = ux.diff(x)
    uy = u.diff(y)
    uyy = uy.diff(y)

    v = v(t,x,y)
    vx = v.diff(x)
    vxx = vx.diff(x)
    vy = v.diff(y)
    vyy = vy.diff(y)

    ut = u.diff(t)
    utt = ut.diff(t)


    right1 = nu * uxx + nu * uyy
    right2 = nu * vxx + nu * vyy

    left1 = ut + cx*u*ux + cy*v*uy
    left2 = ut + cx*u*vx + cy*v*vy

    left1_str = StringIO(str(left1)).readline
    left1_tokens = []
    for idx, t in enumerate(tokenize.generate_tokens(left1_str)):
        if(t.type not in [0, 4]):
            left1_tokens.append(t.string)

    right1_str = StringIO(str(right1)).readline
    right1_tokens = []
    for idx, t in enumerate(tokenize.generate_tokens(right1_str)):
        if(t.type not in [0, 4]):
            right1_tokens.append(t.string)

    left2_str = StringIO(str(left2)).readline
    left2_tokens = []
    for idx, t in enumerate(tokenize.generate_tokens(left2_str)):
        if(t.type not in [0, 4]):
            left2_tokens.append(t.string)

    right2_str = StringIO(str(right2)).readline
    right2_tokens = []
    for idx, t in enumerate(tokenize.generate_tokens(right2_str)):
        if(t.type not in [0, 4]):
            right2_tokens.append(t.string)

    # Do we even need the initial condition?
    # This leaves the forcing term and the viscosity as things to modify.
    all_tokens = []
    all_tokens.extend(left1_tokens)
    all_tokens.append("=")
    all_tokens.extend(right1_tokens)
    all_tokens.append("&")
    all_tokens.extend(left2_tokens)
    all_tokens.append("=")
    all_tokens.extend(right2_tokens)

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


def simulate(cx, cy, SAVE_STEPS, TOTAL_TIME):
    #print("CX: {}, CY: {}, NU: {}".format(cx, cy, nu))
    #cx, cy = 3, -3
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    #raise
    #nu = 0.01
    #nu = 0.01

    sig = 0.0001*nu/(dx*dy)
    #sigma = 0.0009
    dt = sig * dx * dy / nu
    dt = TOTAL_TIME/SAVE_STEPS
    #print("dt: {}".format(dt))
    nt = int(numpy.ceil(TOTAL_TIME/dt))
    #print("TIMESTEPS: {}".format(nt))
    #nt = 10000
    #print("nt: {}".format(nt))
    #print("\nNU: {}".format(nu))
    #print("TOTAL SIM TIME: {}".format(int(dt * nt)))
    #print("X STABILITY: {0:.4f} {1}".format(numpy.abs(cx * dx / (2*nu)), numpy.abs(cx*dx / (2*nu)) < 1.))
    #print("Y STABILITY: {0:.4f} {1}".format(numpy.abs(cy * dy / (2*nu)), numpy.abs(cy*dy / (2*nu)) < 1.))
    #print("TOTAL STABILITY: {0:.4f} {1}".format(numpy.abs((cy**2 + cx**2)**(0.5) * dx / (2*nu)),
    #                                            numpy.abs((cy**2 + cx**2)**(0.5) * dx / (2*nu) < 1.)))
    #print("TIME STABILITY: {0:.4f} {1}".format(nu*dt/(dx**2), nu*dt/(dx**2) < 0.25))
    #raise

    #print("X STABILITY??: {0:.4f} {1:.4f} {2}".format(numpy.abs(2*nu/(cx)), dx, numpy.abs(2*nu/(cx)) > dx))
    #print("Y STABILITY??: {0:.4f} {1:.4f} {2}".format(numpy.abs(2*nu/(cy)), dy, numpy.abs(2*nu/(cy)) > dy))
    #print("TIME STABILITY: {0:.4f} {1:.4f} {2}".format((dx*dy)/(2*nu), dt, (dx*dy)/(2*nu) > dt))


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
    distribution = lambda x, x0: 1/(2.*np.pi * sigma**2) * np.exp(-(np.linalg.norm(x-x0, axis=2)**2)/(2.*sigma**2))

    mat = make_spd_matrix(2)
    #print(mat)
    eig = np.linalg.eig(mat)[0][0]
    mat = mat/(20*eig)
    #print(np.linalg.eig(mat)[0])

    #raise
    X, Y = numpy.meshgrid(x, y)
    #print(X.shape, Y.shape)
    X, Y = numpy.meshgrid(x, y)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    grid = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1)

    pad = torch.nn.ReflectionPad2d((1,1,1,1))
    pad_grid = torch.cat((pad(X.unsqueeze(0)).unsqueeze(-1), pad(Y.unsqueeze(0)).unsqueeze(-1)), dim=-1)[0]

    #u = torch.Tensor(distribution(grid, torch.Tensor([0,0])))
    u = scipy.stats.multivariate_normal.pdf(grid, mean=grid[x_start][y_start], cov=mat)
    u = torch.Tensor(u)

    # Shift distribution
    #u = torch.roll(u, shifts=(x_start, y_start), dims=(0,1))
    #u0 = u.clone()
    #fig = pyplot.figure(figsize=(11, 7), dpi=100)
    #ax = fig.add_subplot(projection='3d')
    #X, Y = numpy.meshgrid(x, y)
    #ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=1, cstride=1)
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    #plt.savefig("./burgers_example.png")
    #plt.show()
    #raise
        

    #fig = pyplot.figure(figsize=(11, 7), dpi=100)
    #ax = fig.add_subplot(projection='3d')
    #X, Y = numpy.meshgrid(x, y)
    #ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=1, cstride=1)
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    #plt.show()
    #plt.savefig("./ic_burgers_example.png")
    #raise

    u0 = u.clone()

    #for n in tqdm(range(nt + 1)): ##loop across number of time steps
    all_us = torch.empty((100, 128, 128))
    all_vs = torch.empty((100, 128, 128))
    times = torch.empty(100)
    #for n in tqdm(range(nt + 1)): ##loop across number of time steps
    #for n in range(nt + 1): ##loop across number of time steps
    for n in range(1, nt + 1): ##loop across number of time steps
        un = u.clone()

        # Calculate finite differences
        un = u.clone()
        #vn = v.clone()

        x_adv = torch.Tensor([-dt*n*cx]).float()
        y_adv = torch.Tensor([-dt*n*cy]).float()

        new_X, new_Y = numpy.meshgrid(x, y)
        new_X = torch.Tensor(new_X)
        new_Y = torch.Tensor(new_Y)
        #print(dt, n, cx)
        new_X += x_adv * torch.ones((X.shape))
        new_Y += y_adv * torch.ones((Y.shape))

        #pad1 = torch.nn.ReflectionPad1d((1,1))
        new_grid = torch.cat((pad(new_X.unsqueeze(0)).unsqueeze(-1), pad(new_Y.unsqueeze(0)).unsqueeze(-1)), dim=-1)[0]

        # Need to figure this out...
        new_u = torch.Tensor(scipy.stats.multivariate_normal.pdf(new_grid, mean=(new_grid[x_start][y_start]), cov=mat))

        # Periodic new grid
        new_grid = ((new_grid+1)%2)-1

        new_u = scipy.interpolate.griddata(
                   new_grid.reshape((-1,2)),
                   new_u.reshape((-1,1)),
                   pad_grid.reshape((-1,2)),
                   method='nearest'
        )
        #print(new_grid.shape)
        new_u = torch.Tensor(new_u.reshape((130,130))[1:-1, 1:-1])
        if(new_u.isnan().any()):
            raise ValueError("Something is still nan")

        den = 1#int(nt/10)

        all_us[n-1] = new_u.clone()
        times[n-1] = TOTAL_TIME*(n)/nt

        #raise
        #u = None

        # Calculate finite differences for diffusion term


    #raise
    x = numpy.linspace(-1, 1, nx)
    y = numpy.linspace(-1, 1, ny)
    X, Y = numpy.meshgrid(x, y)
    X = torch.tensor(X)
    Y = torch.tensor(Y)

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
    v0 = u0.clone()
    all_vs = all_us.clone()

    return u0, all_us, v0, all_vs, torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1), times



def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 10
    nt = 100
    inits = 200
    cxs = numpy.arange(-2, 2, 1.)
    cys = numpy.arange(-2, 2, 1.)
    #cxs = numpy.arange(-1, 1, 1.)
    #cys = numpy.arange(-1, 1, 1.)
    #h5f = h5py.File("../2D_NS_DATA/gaussian_2d_advdiff_{}s_{}inits_{}systems_match_ns.h5".format(TOTAL_TIME, inits,
    h5f = h5py.File("../pde_data/gaussian_2d_advdiff_{}s_{}inits_{}systems_match_ns.h5".format(TOTAL_TIME, inits,
                                                        len(cxs)*len(cys)), 'w')

    #for nu in nus[::-1]:
    for cx in cxs:
        for cy in cys:
            print("CX: {0:.4f}, CY: {1:.4f}, TIMESTEPS: {2}".format(cx, cy, nt))

            for i in tqdm(range(inits)):
                key = 'ad_{}_{}_{}_{}'.format(nu, cx, cy, i)
                u0, u, v0, v, grid, times = simulate(cx, cy, SAVE_STEPS, TOTAL_TIME)

                dataset = h5f.create_group(key)
                dataset['coeffs'] = [cx, cy]
                dataset['u0'] = u0
                dataset['u'] = u
                dataset['v0'] = v0
                dataset['v'] = v
                dataset['nu'] = 0
                dataset['grid'] = grid
                dataset['time'] = times
                #raise


if __name__ == '__main__':
    main()

