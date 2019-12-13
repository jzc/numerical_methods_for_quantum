import numpy as np
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.optimize import newton


pi = np.pi
hbar = (1/(2*pi))
m_e = 1
eV = 1.60218e-19

def same_sign(a, b):
    return (a < 0) == (b < 0)

def count_nodes(x):
    n = 0
    for a, b in zip(x[1:], x[2:-1]):
        if not same_sign(a, b):
            n += 1
    return n

def bisect(f, a, b, epsilon=1e-6):
    fa = f(a)
    fb = f(b)
    while True:
        c = (a+b)/2
        if b-a < epsilon:
            return c
        
        fc = f(c)
        if not same_sign(fa, fc):
            a = c
            fa = fc
        else: 
            b = c
            
def normalize(x, psi):
    A = scipy.integrate.simps(psi**2, x)
    return psi/np.sqrt(A) 

def solve_tise(E, V, xmin, xmax, s=1e-2, max_step=np.inf):
    def f(t, y):
        return np.array([
            y[1],
            -2*m_e/hbar**2*(E-V(t))*y[0]
        ])
    
    res = scipy.integrate.solve_ivp(f, (xmin, xmax), (0, s), max_step=max_step)
    return res

def solve_energies(V, xmin, xmax, n=100, ret_evs=False):
    x, dx = np.linspace(xmin, xmax, n, retstep=True)
    k = hbar**2/(2*m_e*dx**2)
    Vx = np.vectorize(V)(x)
    Es, Vs = scipy.linalg.eigh_tridiagonal(np.ones(n-2)*2*k+Vx[1:-1], np.ones(n-3)*-k)
    return (Es, Vs) if ret_evs else Es

def ISW_energy(n, L):
    return (n**2*pi**2*hbar**2)/(2*m_e*L**2)

def shoot(Emin, Emax, V, xmin, xmax):
    def g(E):
        res = solve_tise(E, V, xmin, xmax)
        return res.y[0,-1]
    
    return bisect(g, Emin, Emax)

def match(E0, E1, V, xmin, xmax, xmatch, epsilon=1e-6):
    def g(E):
        left_shot = solve_tise(E, V, xmin, xmatch)
        right_shot = solve_tise(E, V, xmax, xmatch)
        C = left_shot.y[0,-1]/right_shot.y[0,-1]
        slope_diff = left_shot.y[1, -1]-C*right_shot.y[1, -1]
        return slope_diff
    
    E, res = newton(g, x0=E0, x1=E1, disp=False, full_output=True)
    print(f"converged: {res.converged}")
    return E

def splice(E, V, xmin, xmax, xmatch, max_step=.001):
    left_shot = solve_tise(E, V, xmin, xmatch, max_step=.001)
    right_shot = solve_tise(E, V, xmax, xmatch, max_step=.001)
    C = left_shot.y[0,-1]/right_shot.y[0,-1]
    
    psi = np.concatenate([left_shot.y[0], C*right_shot.y[0, -2::-1]])
    x = np.concatenate([left_shot.t, right_shot.t[-2::-1]])
    return x, psi

def x_axis(ax):
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [0, 0], color="black", alpha=.5, ls="--")
    
def hline(ax, xs):
    ymin, ymax = ax.get_ylim()
    for x in xs:
        ax.plot([x, x], [ymin, ymax], color="gray", alpha=.5, ls="--")