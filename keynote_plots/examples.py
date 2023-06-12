from functools import partial

from tqdm import trange
import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from plotting.quantiles import plot_quantiles


eps = 0             # Additive noise on samples
num_samples = 10    # Number of samples for reconstruction
max_dim = 100       # Maximal number of basis functions
ls_dim = 15         # Number of basis functions for least squares approximation
trials = 10_000     # Number of of trials under which we reconstruct the function

basis = ['legendre', 'chebyshev', 'fourier'][0]
distribution = ['uniform', 'chebyshev', 'cm', 'sp'][0]


limit_weight_sequence = {
    'legendre': np.sqrt(2*np.arange(max_dim)+1),
    'chebyshev': np.sqrt(2/(np.eye(1,max_dim)[0]+np.full(max_dim,1))),
    'fourier': np.full(max_dim, np.sqrt(2))
}

function = lambda x: 1/(1 + 25*x**2)                           # Runge's function
# def weierstrass(a=1/2, b=15, N=10):
#     def w(x):
#         y = np.zeros(x.shape)
#         for n in range(N):
#             y += a**n*np.cos(b**n*np.pi*x)
#         return y
#     return w
# e = lambda k: np.sqrt(2*k+1)*np.eye(1, max_dim, k)[0]
# function = abs                                                 # |x| (not smooth)
# function = np.sign                                             # sing(x) (not continuous)
# function = weierstrass(N=2)                                    # Weierstrass function (nowhere differentiable)
# function = lambda x: np.sqrt(2*1+1)*abs(x) + legval(x, e(40))  # (not smooth & components with large sparsity weight)

def fourier_basis(xs, dim):
    C = np.cos(np.pi*np.arange(dim//2+dim%2)[None]*xs[:,None])
    S = np.sin(np.pi*np.arange(dim//2)[None]*xs[:,None])
    return np.sqrt(2)*np.block([C, S])

evaluate = {
    'legendre': lambda xs, coeffs: legval(xs, np.sqrt(2*np.arange(len(coeffs))+1)*coeffs).T,
    'chebyshev': lambda xs, coeffs: chebval(xs, np.sqrt(2/(np.eye(1,len(coeffs))[0]+np.full(len(coeffs),1)))*coeffs).T,
    'fourier': lambda xs, coeffs: fourier_basis(xs, len(coeffs)) @ coeffs
}

samples = {
    'uniform': lambda num: 2*np.random.rand(num)-1,                # density = lambda x: 1/2
    'chebyshev': lambda num: np.cos(2*np.pi*np.random.rand(num)),  # density = lambda x: 1/np.sqrt(1-x**2)
}


def l1_minimization(A, y, w, weight_sequence=None, shift=0):
    # NOTE: Since we have fever sample points than dimensions, we can interpolate exactly.
    #       It remains to find the interpolating function with minimal l1-norm.
    # A = A[:, :200]  # do not use too many basis functions
    assert A.shape[0] <= A.shape[1]
    if weight_sequence is None:
        weight_sequence = np.ones(A.shape[1])
    x = cp.Variable(A.shape[1])
    # objective = cp.Minimize(weight_sequence@cp.abs(x))
    objective = cp.Minimize(weight_sequence@cp.abs(shift-x))
    constraints = [cp.norm(A@x - y, 2) <= eps]  # eps is global
    prob = cp.Problem(objective, constraints)
    prob.solve()  # Ignore the optimal objective value.
    return x.value  # Return the optimal value for x.

def exact_inversion(A, y, w):
    return np.linalg.lstsq(A[:,:num_samples], y, rcond=None)[0]

def least_squares(A, y, w):
    return np.linalg.lstsq(A[:,:ls_dim], y, rcond=None)[0]


if __name__ == "__main__":
    from contextlib import contextmanager
    @contextmanager
    def plot(name):
        print(f"Plotting: {name}")
        fig = plt.figure(frameon=False, dpi=300)
        fig.set_size_inches(3, 3)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        fig.add_axes(ax)
        yield fig, ax
        fig.savefig(f"{name}.png", bbox_inches='tight', transparent=True, pad_inches=0.1, dpi=300)
        plt.close(fig)


    def plot_approximations(generate_samples, evaluate_basis, evaluate_function, compute_weights, reconstruct, ax, max_dim=100, num_samples=30, trials=10_000):
        C0 = (50 / 255, 116 / 255, 181 / 255)
        C1 = "white"
    
        xs = np.linspace(-1, 1, 2000)
        fxs = evaluate_function(xs)
    
        yss = []
        for trial in trange(trials):
            while True:
                try:
                    x = generate_samples(num_samples)                   # generate sample points
                    A = evaluate_basis(x, np.eye(max_dim))              # generate sampling matrix
                    y = evaluate_function(x)                            # evaluate test function
                    w = compute_weights(x)                              # compute weights
                    coeffs = reconstruct(A, y, w)
                    yss.append(evaluate_basis(xs, coeffs))
                    break
                except cp.error.SolverError:
                    pass
        yss = np.array(yss)
    
        # plot_quantiles(xs, yss, color=C1, axes=ax, zorder=2)
        plot_quantiles(xs, yss, qrange=(0.15,0.85), num_quantiles=5, linewidth=1, color=C1, axes=ax, zorder=2)
        # plot_quantiles(xs, yss, qrange=(0.15,0.85), num_quantiles=5, color=C1, axes=ax, zorder=2)
        # plot_quantiles(xs, yss, qrange=(0, 1), num_quantiles=5, linewidth=1, color=C1, axes=ax, zorder=2)
    
        ax.plot(xs, fxs, linestyle=(0,(0.25,1.5)), color=C0, linewidth=3, dash_capstyle='round', zorder=9)
        ymin = np.min(fxs)-(np.max(fxs)-np.min(fxs))/4
        ymax = np.max(fxs)+(np.max(fxs)-np.min(fxs))/4
        ax.set_xlim(-1,1)
        ax.set_ylim(ymin, ymax)
    
        errors = np.max(abs(yss - fxs[None]), axis=1)
        print(f"Mean error: {np.mean(errors):.2e}")


    weight_function = lambda x: np.ones_like(x, dtype=float)

    # with plot("exact_inversion") as (f,ax):
    #     plot_approximations(samples[distribution], evaluate[basis], function, weight_function, exact_inversion, ax, max_dim=max_dim, num_samples=num_samples, trials=trials)

    # with plot("least_squares") as (f,ax):
    #     plot_approximations(samples[distribution], evaluate[basis], function, weight_function, least_squares, ax, max_dim=max_dim, num_samples=num_samples, trials=trials)

    with plot("standard_l1") as (f,ax):
        plot_approximations(samples[distribution], evaluate[basis], function, weight_function, l1_minimization, ax, max_dim=max_dim, num_samples=num_samples, trials=trials)

    weight_sequence = limit_weight_sequence[basis]
    with plot("weighted_l1") as (f,ax):
        plot_approximations(samples[distribution], evaluate[basis], function, weight_function, partial(l1_minimization, weight_sequence=weight_sequence), ax, max_dim=max_dim, num_samples=num_samples, trials=trials)
