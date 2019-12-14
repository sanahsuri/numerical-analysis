import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import pandas as pd
import time
import mpl_toolkits.mplot3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Implementation of 1D heat equation using Crank-Nicolson method

# forward substitution using lower triangular matrix
def forward_sub(l, b, x):
    n = int(math.sqrt(l.size))
    for i in range(0, n):
        for j in range(0, i):
            b[i] = b[i] - l[i][j] * x[j]
        x[i] = b[i]/l[i][i]
    return x

# LU factorization
def lu_fac(m):
    n = int(math.sqrt(m.size))
    u = np.zeros(shape = (n, n))
    for s in range(0, n):
       for t in range(0, n):
           u[s][t] = m[s][t]

    l = np.zeros(shape = (n, n))
    for j in range(0, n):
        if np.absolute(m[j][j]) == 0:
            return "error!!"
        else:
            for i in range(j+1, n):
                mult = m[i][j]/m[j][j]
                if (j < i):
                    l[i][j] = mult
                for k in range(j+1, n):
                    u[i][k] = m[i][k] - (mult * m[j][k])
    for p in range(0, n):
        for q in range(0, n):
          if (q < p):
              u[p][q] = 0
          if (p == q):
              l[p][q] = 1
    return u, l

def gaus_seidel(m, b, x0, k):
    n = int(math.sqrt(m.size))
    u, l = lu_fac(m)
    for i in range(0, n):
        for j in range(0, n):
            if (i == j):
                u[i][j] = 0
    for i in range(0, k):
        y = b - np.dot(u, x0)
        x0 = forward_sub(l, y, x0)
    return x0

# [xl, xr] -> space interval
# [yb, yt] -> time interval
# M -> steps in space direction
# N -> steps in time direction
# D -> diffusion coefficient
# f -> function
# l -> boundary condition
# r -> boundary condition
# Awj = Bwj-1 + sigma(sj-1 + sj)
def crank_nicolson(xl, xr, yb, yt, M, N, D, f, l, r):
    dx = (xr-xl)/M
    dt = (yt-yb)/N
    sigma = D*dt/(dx*dx)
    m = M-1
    n = N
    a = 2 * np.diag(np.ones(m)) + 2 * sigma * np.diag(np.ones(m)) + (-1 * sigma) * np.diag(np.ones(m-1), 1) + (-1 * sigma) * np.diag(np.ones(m-1), -1)
    b = 2 * np.diag(np.ones(m)) - 2 * sigma * np.diag(np.ones(m)) + sigma * np.diag(np.ones(m-1), 1) + sigma * np.diag(np.ones(m-1), -1)
    lside = np.zeros(n+1, )
    rside = np.zeros(n+1, )
    for i in range(0, n+1):
        lside[i] = l(yb + i*dt)
        rside[i] = r(yb + i*dt)
    sol = np.zeros((m, n+1))
    for i in range(0, m):
        sol[i][0] = f(xl + (i+1)*dx)
    for j in range(1, n+1):
        sides = np.zeros(m)
        sides[0] = lside[j-1] + lside[j]
        sides[m-1] = rside[j-1] + rside[j]
        r = np.matmul(b, (sol[:, j-1])) + sigma*sides
        sol[:, j] = np.linalg.solve(a, r)
    sol = np.vstack((sol, rside))
    sol = np.vstack((lside, sol))
    return sol;

def crank_nicolson_anim(xl, xr, yb, yt, M, N, D, f, l, r, zarray):
    dx = (xr-xl)/M
    dt = (yt-yb)/N
    sigma = D*dt/(dx*dx)
    m = M-1
    n = N
    a = 2 * np.diag(np.ones(m)) + 2 * sigma * np.diag(np.ones(m)) + (-1 * sigma) * np.diag(np.ones(m-1), 1) + (-1 * sigma) * np.diag(np.ones(m-1), -1)
    b = 2 * np.diag(np.ones(m)) - 2 * sigma * np.diag(np.ones(m)) + sigma * np.diag(np.ones(m-1), 1) + sigma * np.diag(np.ones(m-1), -1)
    lside = np.zeros(n+1, )
    rside = np.zeros(n+1, )
    for i in range(0, n+1):
        lside[i] = l(yb + i*dt)
        rside[i] = r(yb + i*dt)
    sol = np.zeros((m, n+1))
    for i in range(0, m):
        sol[i][0] = f(xl + (i+1)*dx)
    for j in range(1, n+1):
        sides = np.zeros(m)
        sides[0] = lside[j-1] + lside[j]
        sides[m-1] = rside[j-1] + rside[j]
        r = np.matmul(b, (sol[:, j-1])) + sigma*sides
        sol[:, j] = np.linalg.solve(a, r)
        temp = np.vstack((sol, rside))
        temp = np.vstack((lside, temp))
        zarray[:,:,j] = temp
    return zarray;

def f(x):
    #return 10
    return math.sin(2 * math.pi * x) * math.sin(2 * math.pi * x)
    #return np.exp(-0.5 * x)


def p(x):
    return 0;
    #return 10;
    return np.exp(x)

def q(x):
    return 0;
    #return 10;
    return np.exp(x - 0.5)

def do_anim(xl, xr, yb, yt, M, N, D, f, l, r):
    frn = M+1 # frame number of animation
    fps = 10 # frame per sec
    x = np.linspace(-1, 1, M+1)
    t = np.linspace(-1, 1, N+1)
    t, x = np.meshgrid(t, x)
    zarray = np.zeros((M+1, N+1, frn))

    zarray = crank_nicolson_anim(xl, xr, yb, yt, M, N, D, f, l, r, zarray)
    max = np.amax(zarray)
    min = np.amin(zarray)
    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, t, zarray[:,:,frame_number], cmap="magma")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(x, t, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(min - 0.2 * min, max + 0.2 * max)
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=1000/fps)

    fn = 'plot_surface_animation_funcanimation'
    ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    plt.show()

def do_static(xl, xr, yb, yt, M, N, D, f, l, r):
    x = np.linspace(-1, 1, M+1)
    t = np.linspace(-1, 1, N+1)
    t, x = np.meshgrid(t, x)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    sol = crank_nicolson(xl, xr, yb, yt, M, N, D, f, l, r)
    ax.plot_surface(x, t, sol, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('temp');
    plt.show()

do_anim(0, 1, 0, 1, 100, 100, 1, f, p, q)
#do_static(0, 1, 0, 1, 100, 100, 4, f, p, q)
