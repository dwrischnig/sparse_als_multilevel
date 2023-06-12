import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def plot_matrix(alpha):
    x = y = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + (alpha*Y)**2
    
    fig = plt.figure(frameon=False)
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
    fig.add_axes(ax)
    cs = ax.contour(X, Y, Z, 6, colors='white')
    return fig, ax, cs

fig, ax, cs = plot_matrix(1)
fig.savefig('kernel_0.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.close(fig)

fig, ax, cs = plot_matrix(2)
fig.savefig('kernel_1.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.close(fig)

fig, ax, cs = plot_matrix(1e6)
fig.savefig('kernel_2.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.close(fig)
