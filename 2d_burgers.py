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


def simulate(nu, cx, cy, SAVE_STEPS, TOTAL_TIME, idx):

    np.random.seed(idx)
    torch.manual_seed(idx)

    #print("CX: {}, CY: {}, NU: {}".format(cx, cy, nu))
    #cx, cy = 3, -3
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    #nu = 0.01
    #nu = 0.01

    #sigma = 0.0025*nu/(dx*dy)
    sigma = 0.0002*nu/(dx*dy)
    #print(sigma)
    dt = sigma * dx * dy / nu
    #print(dt)
    #nt = 36000
    #print("dt: {}".format(dt))
    nt = int(TOTAL_TIME/dt)
    SAVE_EVERY = nt//SAVE_STEPS + 1
    #print("nt: {}".format(nt))
    #print("SAVE_EVERY: {}".format(SAVE_EVERY))
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
    u = scipy.stats.multivariate_normal.pdf(grid, mean=grid[x_start][y_start], cov=mat)
    v = scipy.stats.multivariate_normal.pdf(grid, mean=grid[x_start][y_start], cov=mat)
    u = torch.Tensor(u)
    v = torch.Tensor(v)
    u0 = u.clone()
    v0 = v.clone()
    ###########################################

    #for n in tqdm(range(nt + 1)): ##loop across number of time steps
    all_us = torch.empty((SAVE_STEPS, 128, 128))
    all_vs = torch.empty((SAVE_STEPS, 128, 128))
    times = torch.empty(SAVE_STEPS)
    for n in range(nt + 1): ##loop across number of time steps

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

        #print(adv_u.shape)
        #print(adv_v.shape)

        # Calculate update
        u = nu*dt*diff_u/dx**2 - dt*adv_u/dx + u
        v = nu*dt*diff_v/dy**2 - dt*adv_v/dy + v

        if((n+1)%SAVE_EVERY == 0):
            all_us[(n)//SAVE_EVERY] = u.clone()
            all_vs[(n)//SAVE_EVERY] = v.clone()
            times[(n)//SAVE_EVERY] = np.round(TOTAL_TIME*(n+1)/nt, 3)

    #print(times)
    #raise
    #X, Y = numpy.meshgrid(x, y)

    #fig = pyplot.figure(figsize=(11, 7), dpi=100)
    #ax = fig.add_subplot(projection='3d')
    #X, Y = numpy.meshgrid(x, y)
    #ax.plot_surface(X, Y, u0, cmap=cm.viridis, rstride=1, cstride=1)
    #ax.set_xlabel('$x$')
    #ax.set_ylabel('$y$')
    ##plt.savefig("./burgers_example.png")
    #plt.show()
    #print(all_us)

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
    if(all_us.isnan().any()):
        raise ValueError("Issue in Simulation")

    return u0, all_us, v0, all_vs, torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1), times


    #dx = 1 / (nx - 1)
    #dy = 1 / (ny - 1)
    ###set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    ##v[int(y_start / dy):int(y_end / dy + 1),int(x_start / dx):int(x_end / dx + 1)] = 2
    #u0 = u.clone()
    #
    ####(plot ICs)
    ##fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ##ax = fig.add_subplot(projection='3d')


    #X, Y = numpy.meshgrid(x, y)
    #X = torch.tensor(X)
    #Y = torch.tensor(Y)
    #return u0, all_us, torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1), times
            

def main():
    SAVE_STEPS = 100
    TOTAL_TIME = 10
    nt = 10000
    inits = 10
    #nus = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    nus = []
    #for j in [-9, -8, -7, -6, -5]:
    #for j in [-8, -7, -6]:
    for j in [-5, -4]:
        for i in numpy.arange(1, 10, 1):
            nus.append(float("{0}e{1}".format(i, j)))
    nus.append(1e-3)
    cxs = numpy.arange(-2, 2, 1.)
    cys = numpy.arange(-2, 2, 1.)
    h5f = h5py.File("../2D_NS_DATA/gaussian_2d_Burgers_{}s_{}inits_{}systems_ns_match.h5".format(TOTAL_TIME, inits,
                                                        len(nus)*len(cxs)*len(cys)), 'w')

    idx = 0
    for nu in nus:
        for cx in cxs:
            for cy in cys:
                print("NU: {0:.4f}, CX: {1:.4f}, CY: {2:.4f}, TIMESTEPS: {1}".format(nu, cx, cy, nt))

                for i in tqdm(range(inits)):
                    key = 'Burgers_{}_{}_{}_{}'.format(nu, cx, cy, i)
                    u0, u, v0, v, grid, times = simulate(nu, cx, cy, SAVE_STEPS, TOTAL_TIME, idx)

                    raw_tokens = get_tokens(nu, cx, cy)
                    tokens = encode_tokens(raw_tokens)

                    dataset = h5f.create_group(key)
                    dataset['tokens'] = tokens
                    dataset['u0'] = u0
                    dataset['u'] = u
                    dataset['v0'] = v0
                    dataset['v'] = v
                    dataset['nu'] = nu
                    dataset['grid'] = grid
                    dataset['time'] = times
                    #raise


if __name__ == '__main__':
    main()
