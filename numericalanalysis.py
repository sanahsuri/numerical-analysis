import numpy as np
import math

# returns polynomial wrt x and coefficients in c
def poly(x, c = [-1, 1, 0, 1]):
    return np.polynomial.polynomial.polyval(x, c)

def func(x):
    return np.power(1-x, 0.33)

# bisection method to find roots
# f is a function, [a, b] is the interval and e is the tolerance
def bisection_method(f, a, b, e):
    if f(a) * f(b) > 0:
        return "error"
    while ((b-a)/2 > e):
        c = (a+b)/2
        if f(c) == 0.0:
             return c
        elif f(a) * f(c) < 0.0:
            b = c;
        elif f(b) * f(c) < 0.0:
            a = c;
    return c

#print(poly(1))
#print(bisection_method(poly, 0, 1, 0.0005))

# fixed point iteration
# g is the function, x0 is initial guess, k is iterations
def fpi(g, x0, k):
    for i in range(0, k):
        x0 = g(x0)
    return x0;

#print(fpi(func, 0.5, 25))

p2 = np.poly1d([1, 0, 1, -1])
p2d = np.polyder(p2)
#print(p2)

# g is in the form of a poly1d
def newton(g, x0, k):
    gd = np.polyder(g)
    for i in range(0, k):
        x0 = x0 - (g(x0)/gd(x0))
    return x0

#print("newtons:")
#print(newton(p2, -0.7, 7))

def secant(g, x0, x1, k):
    for i in range(0, k):
        y = x1 - (g(x1) * (x1 - x0))/(g(x1) - g(x0))
        x0 = x1
        x1 = y
    return x1

#print("secant:")
#print(secant(p2, 0, 1, 9))

# not working right
def false_pos(g, a, b, k):
    for i in range(0, k):
        c = (b * g(a)) - (a * g(b))/(g(a) - g(b))
        if g(c) == 0:
            return c
        elif g(a) * g(b) < 0:
            b = c
        else:
            a = c
    return c

#A = np.array([1, 2, 3], [1, 2, 3])
#a = np.array([[1, 2, 3], [1, 2, 3]])
#print(int(a.size/a[0].size))

# Gaussian elimination
def gaus_elim(m, b):
    n = int(math.sqrt(m.size))
    for j in range(0, n):
        if np.absolute(m[j][j]) == 0:
            return "error!!"
        else:
            for i in range(j+1, n):
                mult = int(m[i][j]/m[j][j])
                for k in range(j+1, n):
                    m[i][k] = m[i][k] - (mult * m[j][k])
                b[i] = b[i] - (mult * b[j])
    for p in range(0, n):
        for q in range(0, n):
            if (q < p):
                m[p][q] = 0
    # Back substitution step
    x = [0.0]*n
    for r in range(n-1, -1, -1):
        for s in range(r+1, n):
            b[r] = b[r] - m[r][s] * x[s]
        x[r] = b[r]/m[r][r]
        print(x[r])
    return x

#m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#b = np.array([3.0, 3.0, -6.0])
m = np.array([[1.0, 2.0], [3.0, 4.0]])
l = np.array([[0.0, 0.0], [0.0, 0.0]])
b = np.array([3, 2])
#print(gaus_elim(m, b))

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

#l = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
m1, l1 = lu_fac(m)
#print(m1)
#print(l1)
ml = np.dot(m1, l1)
#print(ml)

# PA = LU

# Jacobi method

def jacobi_method_book(m, b, x0, k):
    n = int(math.sqrt(m.size))
    u, l = lu_fac(m)
    print(u)
    print(l)
    d_in = np.zeros(shape = (n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (i == j):
                d_in[i][j] = 1/m[i][j]
                u[i][j] = 0.0
                l[i][j] = 0.0
    for p in range(0, k):
        print(p)
        x0 = np.dot(d_in, (b1 - np.dot(l+u, x0)))
    return x0

def jacobi_method(m, b, x0, k):
    n = int(math.sqrt(m.size))
    d_in = np.zeros(shape = (n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (i == j):
                d_in[i][j] = 1/m[i][j]
                m[i][j] = 0
    for l in range(0, k):
        x0 = np.dot(d_in, (b - np.dot(m, x0)))
    return x0;

# backward substitution using upper triangular matrix
def backward_sub(u, b, x):
    n = int(math.sqrt(u.size))
    for i in range(n-1,-1,-1):
        s = b[i]
        for j in range(i+1,n):
            b[i] = b[i] - u[i][j] * x[j]
        x[i] = b[i]/u[i][i]
    return x

# forward substitution using lower triangular matrix
def forward_sub(l, b, x):
    n = int(math.sqrt(l.size))
    for i in range(0, n):
        for j in range(0, i):
            b[i] = b[i] - l[i][j] * x[j]
        x[i] = b[i]/l[i][i]
    return x


mat = np.array([[3.0, 1.0, -1.0], [2.0, 4.0, 1.0], [-1.0, 2.0, 5.0]])
b1 = np.array([4.0, 1.0, 1.0])
x0 = np.array([0.0, 0.0, 0.0])


#mat = np.array([[3.0, 1.0], [1.0, 2.0]])
#b1 = np.array([5.0, 5.0])
#x0 = np.array([0.0, 0.0])
#print(jacobi_method(mat, b1, x0, 40))
#print(np.dot(l, u))

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

print(gaus_seidel(mat, b1, x0, 100))
