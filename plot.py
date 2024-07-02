import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, MaxNLocator
from OppOpPopInit import OppositionOperators
from utils.utils_funcs import get_good_arrow_place

def plot_3d(func, points_by_dim=50, title='', bounds=None, show_best_if_exists=True,
            save_as=None, cmap='twilight', plot_surface=True, plot_heatmap=True, optimization_paths=None):

    assert plot_surface or plot_heatmap, "Should plot at least surface or heatmap!"

    if bounds is None:
        bounds = func.bounds

    xmin, xmax, ymin, ymax = bounds

    # 50 Points at each dimension by default
    x = np.linspace(xmin, xmax, points_by_dim)
    y = np.linspace(ymin, ymax, points_by_dim)

    a, b = np.meshgrid(x, y)

    data = np.empty((points_by_dim, points_by_dim))
    for i in range(points_by_dim):
        for j in range(points_by_dim):
            data[i, j] = func(np.array([x[i], y[j]]))

    a = a.T
    b = b.T

    # Check bounds to each dimension
    l_a, r_a, l_b, r_b = xmin, xmax, ymin, ymax
    l_c, r_c = data.min(), data.max()
    levels = MaxNLocator(nbins=15).tick_values(l_c, r_c)

    # Create figure
    fig = plt.figure(figsize=(12, 10))

    # Initialize ax1 and ax2
    ax1, ax2 = None, None

    if plot_heatmap and plot_surface:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    elif plot_heatmap:
        ax1 = fig.add_subplot(1, 1, 1)
    elif plot_surface:
        ax2 = fig.add_subplot(1, 1, 1, projection='3d')

    title = r"$\bf{" + title + r"}$"
    min_title = title[::]

    if plot_heatmap:
        c = ax1.contourf(a, b, data, cmap=cmap, levels=levels, vmin=l_c, vmax=r_c)
        ax1.set_title(title, fontsize=15)
        ax1.axis([l_a, r_a, l_b, r_b])
        # fig.colorbar(c, ax=ax1)

    if plot_surface:
        surf = ax2.plot_surface(a, b, data, cmap=cmap, linewidth=0, antialiased=False)
        ax2.set_xlabel('first dim', fontsize=10)
        ax2.set_ylabel('second dim', fontsize=10)
        ax2.set_zlim(l_c, r_c)
        ax2.zaxis.set_major_locator(LinearLocator(4))

        # if not plot_heatmap:
        #     # fig.colorbar(surf, ax=ax2)

        ax2.contour(a, b, data, zdir='z', offset=0, cmap=cmap)
        ax2.view_init(60, 35)
        ax2.set_title(min_title, fontsize=15, loc='right')

    if optimization_paths:
        for path, label, color in optimization_paths:
            if plot_heatmap:
                ax1.plot(path[:, 0], path[:, 1], color=color, label=label)
                ax1.legend(loc='lower right')

    if func.x_best is not None:
        title += f"\n best solution: f{func.x_best} = {round(func(func.x_best))}"
        if show_best_if_exists and plot_heatmap:
            xytext = get_good_arrow_place(func.x_best, bounds)
            bbox = dict(boxstyle="round", fc="0.8")
            arrowprops = dict(arrowstyle="->", connectionstyle="angle, angleA=0, angleB=90, rad=10")
            ax1.annotate(
                'global minimum',
                xy=tuple(func.x_best),
                xytext=xytext,
                arrowprops=dict(facecolor='red', shrink=0.05),
                bbox=bbox
            )

    fig.tight_layout()
    
    if save_as:
        plt.savefig(save_as)
    
    return fig