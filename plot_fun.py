from matplotlib import cm
from scipy.interpolate import interpn

def density_scatter(x, y, ax=None, sort=True, bins=50, mean_absolute_error=None, rmse=None, res=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    x_eq_y = np.linspace(0, x.max())
    plt.plot(x_eq_y, x_eq_y, color='orange', label='x=y')
    plt.scatter(x, y, c=z)

    sorted_pairs = sorted((i, j) for i, j in zip(x, y))
    x_sorted = []
    y_sorted = []
    for i, j in sorted_pairs:
        x_sorted.append(i)
        y_sorted.append(j)

    # change this to e.g 3 to get a polynomial of degree 3 to fit the curve
    order_of_the_fitted_polynomial = 1
    p30 = np.poly1d(np.polyfit(x_sorted, y_sorted, order_of_the_fitted_polynomial))
    plt.plot(x_sorted, p30(x_sorted), color='red', label='linjär anpassning')

    ax.set_aspect('equal', 'box')
    plt.xlabel("Målvärde [m]")
    plt.ylabel("Prediktion [m]")
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    if mean_absolute_error is not None and rmse is not None and res is not None:
        fig_text = f"MAE={mean_absolute_error:.3f}m\nRMSE={rmse:.3f}m\nR={res:.3f}"
        plt.plot([], [], ' ', label=fig_text)
        # plt.text(0, 2.2, s=s, fontsize=12)

    # norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(), ax=ax)
    cbar.ax.set_ylabel('Densitet')
    ax.legend()
    return ax