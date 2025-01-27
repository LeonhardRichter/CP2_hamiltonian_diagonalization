# %%
import pickle
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from scipy.optimize import curve_fit

from simulate_dicke import sim_dicke

# %%
# from
#   \usepackage{layout}
#   layout
# in latex document
# 0.01384 is the conversion from pt to inches
latex_textwidth = 0.01384 * 434
latex_textheight = 0.01384 * 623


# %%
def plot_sim_dicke(
    N: int,
    T: float,
    n: int = 0,
    freq: float = 1.0,
    coup: float = 1.0,
    time_step=None,
    spin_state_name="excited",
):
    """
    Only for quick simulations.
    """
    tt, vt, et, duration = sim_dicke(
        N=N,
        T=T,
        n=n,
        freq=freq,
        coup=coup,
        time_step=time_step,
        spin_state_name=spin_state_name,
    )
    plt.scatter(tt, np.real_if_close(et[0]), s=10)
    plt.xlabel("t")
    plt.ylabel("$\\langle v(t), \\hat n v(t)\\rangle$")
    plt.title(f"N={N}")
    plt.show()


# %%
# with open("dicke_sim_09_01_2025-16_19_05_N-2-20_T-6.0.pickle", "rb") as file:
#     data = pickle.load(file)

with open(
    "data/dicke_sim_09_01_2025-17_04_45_excited_N-2-60_T-6.0.pickle", "rb"
) as file:
    data_excited_long = pickle.load(file)

with open(
    "data/dicke_sim_10_01_2025-14_46_06_superradiant_N-2-60_T-6.0.pickle", "rb"
) as file:
    data_superradient_long = pickle.load(file)

# the spin state name is missing in the data set
for res in data_excited_long:
    res["spin_state"] = "excited"

with open(
    "data/dicke_sim_13_01_2025-12_59_32_excited_N-2-60_T-3.0.pickle", "rb"
) as file:
    data_excited = pickle.load(file)


with open(
    "data/dicke_sim_14_01_2025-11_24_23_superradiant_N-2-60_T-3.0.pickle", "rb"
) as file:
    data_superradient = pickle.load(file)


# define functions for computing the local gradient
def slope(x1, y1, x2, y2) -> float:
    return (y2 - y1) / (x2 - x1)


def arr_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y1 = y[:-1]
    x1 = x[:-1]
    y2 = y[1:]
    x2 = x[1:]

    return slope(x1, y1, x2, y2)


#################################################################################
# define functions for plotting data
ylabel_dict = {
    "expect": {
        "superradiant": r"$n_{\text{sup}}(t)$",
        "excited": r"$n_{\text{exc}}(t)$",
    },
    "variance": {
        "superradiant": "$\\sigma^2_{\\text{sup}}(t)$",
        "excited": "$\\sigma^2_{\\text{exc}}(t)$",
    },
    "slopes": {
        "superradiant": "$\\frac{\\Delta_{\\text{sup}}(t)}{\\delta}$",
        "excited": "$\\frac{\\Delta_{\\text{exc}}(t)}{\\delta}$",
    },
}


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def plot_data(
    data: list[dict],
    max_N: int = None,
    min_N: int = None,
    N_selection: Iterable = None,
    max_N_color: int = None,
    y: tuple = (None, None),
    x: tuple = (None, None),
    metric: str = "expect",
    set_colorbar: bool = True,
    use_larger_font: bool = False,
    line_thickness: float = 2,
    width=latex_textwidth,
):
    if use_larger_font:
        plt.style.use("mpl_large.style")
    if not use_larger_font:
        plt.style.use("mpl_small.style")

    if metric == "slopes":
        ys = np.array([np.real(res["st"][0]) for res in data])[::-1]
        xs = np.array([res["tt"][:-1] for res in data])[::-1]

    if metric == "variance":
        ys = np.array([np.real(res["et"][1]) for res in data])[::-1]
        xs = np.array([res["tt"] for res in data])[::-1]

    if metric == "expect":
        ys = np.array([np.real(res["et"][0]) for res in data])[::-1]
        xs = np.array([res["tt"] for res in data])[::-1]

    Ns = np.array([res["N"] for res in data])[::-1]

    state_name = data[0]["spin_state"]
    # if plot_slopes:
    #     # change to derivative approximation
    #     ys = np.array([arr_slope(x, y) for x, y in zip(xs, ys)])
    #     xs = np.array([x[:-1] for x in xs])

    min_x, max_x = x
    min_y, max_y = y

    if max_N is None:
        max_N = max(Ns)
    if min_N is None:
        min_N = min(Ns)

    # apriori filter for everything
    filter = np.logical_and(Ns <= max_N, min_N <= Ns)

    xs = xs[filter]
    ys = ys[filter]
    Ns = Ns[filter]

    fig, ax = plt.subplots()
    # fig.set_size_inches(8, 4)
    fig.set_layout_engine("tight")
    w, h = fig.get_size_inches()
    aspectratio = h / w
    fig.set_size_inches(width, width * aspectratio, forward=False)

    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    if max_N_color is None:
        max_N_color = np.max(Ns)

    num_colors = max_N_color - np.min(Ns)

    cmap = truncate_colormap(mpl.cm.viridis, 0.2, 0.92, num_colors)
    colors = cmap(np.linspace(0, 1, num_colors))

    # a posterori filter just for the selection of graphs, colors are unaffected
    if N_selection is not None:
        selection_filter = np.isin(Ns, N_selection)
    if N_selection is None:
        selection_filter = np.ones_like(Ns)

    if metric == "slopes":
        ax.axhline(y=0, xmin=0, xmax=1, color="black")

    for N, x, y, c, f in zip(
        Ns[::-1], xs[::-1], ys[::-1], colors[::-1], selection_filter[::-1]
    ):
        if f:
            ax.plot(
                # x[:: int(len(x) / 500)],
                # y[:: int(len(x) / 500)],
                x,
                y,
                ".-",
                alpha=0.75,
                linewidth=line_thickness * 0.75,
                # markeredgewidth=line_thickness*.5,
                markersize=line_thickness,
                label=f"N={N}",
                color=c,
                rasterized=True,
            )
    ax.set_xlabel("t")
    # ax.set_ylabel("$\\langle v(t), \\hat n v(t)\\rangle$")
    # ax.set_ylabel("$\\langle \\hat n\\rangle_{v(t)}$")
    ax.set_ylabel(ylabel_dict[metric][state_name])

    if max_y is not None and min_y is not None:
        ax.set_ylim(min_y, max_y)
    if max_x is not None and min_x is not None:
        ax.set_xlim(min_x, max_x)
    else:
        ax.set_xlim(np.min(xs[0]), np.max(xs[0]))

    # x_ticks = list(ax.get_xticks())
    # x_ticks.append(xs[0][-1])
    # ax.set_xticks = np.arange(min(xs[0]), max(xs[0]), 0.1)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

    if set_colorbar:
        bounds = np.arange(np.min(Ns), max_N_color + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            # label="N",
            ax=ax,
            orientation="vertical",
            # fraction=0.2,
            aspect=10,
            # ticks=Ns,
        )
        cbar.ax.set_xlabel("N", rotation="horizontal")

        # # always add highest N to ticks
        # cbar_ticks = list(cbar.get_ticks())
        # cbar_ticks.append(Ns[-1])
        # if cbar_ticks[-1] - cbar_ticks[-2] <= 4:
        #     cbar_ticks.pop(-2)
        if np.max(Ns) >= 20:
            cbar_ticks = list(range(10, np.max(Ns), 10))
        cbar_ticks.append(np.max(Ns))
        cbar_ticks.insert(0, np.min(Ns))
        # filter ticks for selected graphs
        if N_selection is not None:
            cbar_ticks = N_selection

        cbar.set_ticks(np.array(cbar_ticks))
    return fig


#################################################################################
# compute local gradients and add to data
for data in [data_excited, data_superradient]:
    for res in data:
        res["st"] = list()
        for et in res["et"]:
            yt = np.real_if_close(np.array(et))
            xt = np.array(res["tt"])
            # change to derivative approximation
            slopes = arr_slope(xt, yt)
            res["st"].append(slopes)

#################################################################################
# visualize full data (fig. 6)
fig_excited = plot_data(
    data_excited,
    # max_N=43,
    # x=(0, 6),
    y=(0, 70),
    width=latex_textwidth * 0.5,
    line_thickness=1,
    # use_larger_font=True,
)

fig_superradiant = plot_data(
    data_superradient,
    # max_N=43,
    # x=(0, 6),
    y=(0, 70),
    width=latex_textwidth * 0.5,
    line_thickness=1,
    # use_larger_font=True,
)

fig_excited.savefig(
    "figures/fig_excited.pdf", bbox_inches="tight", dpi=500, transparent=True
)

fig_excited.savefig(
    "figures/fig_excited.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)

fig_superradiant.savefig(
    "figures/fig_superradiant.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)

fig_superradiant.savefig(
    "figures/fig_superradiant.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)

#################################################################################
# visualize selection of data (fig. 3)
selection = [60, 50, 40, 30, 20, 10, 2]
fig_excited_selection = plot_data(
    data_excited,
    # max_N=43,
    # x=(0, 6),
    y=(0, 70),
    width=latex_textwidth * 0.5,
    N_selection=selection,
    max_N_color=60,
    # use_larger_font=True,
)

fig_superradiant_selection = plot_data(
    data_superradient,
    # max_N=43,
    # x=(0, 6),
    y=(0, 70),
    width=latex_textwidth * 0.5,
    N_selection=selection,
    max_N_color=60,
    # use_larger_font=True,
)

fig_excited_selection.savefig(
    "figures/fig_excited_selection.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
fig_excited_selection.savefig(
    "figures/fig_excited_selection.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)

fig_superradiant_selection.savefig(
    "figures/fig_superradiant_selection.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
fig_superradiant_selection.savefig(
    "figures/fig_superradiant_selection.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)


#################################################################################
# define fit function
def quadratic_linear(x, a, b):
    return a * x**2 + b * x


def quadratic_linear_std(x, a_std, b_std):
    return np.sqrt(a_std**2 * np.abs(x**2) + b_std**2 * np.abs(x))


#################################################################################
# define functin for plotting N dependencies in fig. 5
marker_dict = {"superradiant": "o", "excited": "^"}
label_dict = {
    "avg after first bump": "Avg. after init. phase",
    "first high": "Max. in init. phase",
    "first slope": "Grad. of init. phase",
    "initial slope": "Initial gradient",
    "avg after first bump over N": "Avg. after init. phase/N",
    "first high over N": "Max. in init. phase/N",
    "first slope over N": "Grad. of init. phase/N",
    "initial slope over N": "Initial gradient/N",
}
color_dict = {"superradiant": "limegreen", "excited": "dodgerblue"}  # orangered


def plot_data_N_dependence(
    data_list,
    metric: str,
    inset_metric: str | None = None,
    max_N: int = None,
    min_N: int = None,
    fit_function: Callable | None = None,
    fit_function_std: Callable | None = None,
    initial_fit_parameters: tuple | None = None,
    width=latex_textwidth,
    legend: bool = False,
):
    fig, ax = plt.subplots()
    fig.set_layout_engine("tight")
    w, h = fig.get_size_inches()
    aspectratio = h / w
    fig.set_size_inches(width, width * aspectratio, forward=False)

    for data in data_list:
        Ns = np.array([res["N"] for res in data])[::-1]

        if max_N is None:
            max_N = max(Ns)
        if min_N is None:
            min_N = min(Ns)
        # apriori filter for everything
        # filter = np.logical_and(Ns <= max_N, min_N <= Ns)
        filter = np.full_like(Ns, True, dtype=np.bool)

        y = np.array([np.real_if_close(res[metric][0]) for res in data])[::-1]
        x = np.array([res["N"] for res in data])[::-1]
        state_name = data[0]["spin_state"]

        if inset_metric is not None:
            axinset = ax.inset_axes(
                [0.15, 0.5, 0.5, 0.4],
                xlim=(min_N, max_N),
                ylim=(0, 3),
            )
            yinset = np.array(
                [np.real_if_close(res[inset_metric][0]) for res in data]
            )[::-1]
            axinset.scatter(
                x[filter],
                yinset[filter],
                # c=x[filter],
                # cmap=mpl.cm.viridis,
                color=color_dict[state_name],
                alpha=0.7,
                marker=marker_dict.get(state_name, "."),
                # marker="$f=ma$",
                s=10,
                # s=2000,
                # label=f"{state_name} state",
                zorder=1,
            )
            axinset.set_xlim(-0.3, max_N)
            axinset.set_ylim(0, np.max(yinset) + 0.3)
            axinset.set_yticks(np.linspace(0, np.max(yinset), 4))
            axinset.set_xlabel("N")
            axinset.set_ylabel("over N")

        if fit_function is not None:
            x_data = x[filter]
            y_data = y[filter]
            # sigma = 0.01 * y_data
            popt, pcov = curve_fit(
                fit_function,
                x_data,
                y_data,
                initial_fit_parameters,
                # sigma=sigma,
            )
            pstd = np.sqrt(np.diag(pcov))
            # r = y_data - fit_function(x_data, *popt)
            # chisq = sum((r / sigma) ** 2)
            # x_fit = np.linspace(np.min(x_data), np.max(x_data), 1000)
            x_fit = x_data
            y_fit = fit_function(x_fit, *popt)
            ax.plot(
                x_fit,
                y_fit,
                alpha=0.6,
                linewidth=1,
                # color=color_dict[state_name],
                color="black",
                # label="{}*(x-{})^2 + {}*x + {}".format(*np.round(popt, 2)),
                # label=f"{fit_function.__name__} fit",
                # zorder=-1,
            )
            print(metric, ", ", state_name)
            print(f"{fit_function.__name__} fit:")
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            for a, p, err in zip(alphabet, popt, pstd):
                print(
                    f"{a} = \\num\u007b{uncertainties.ufloat(p, err):0.2ue}\u007d"
                )

            if fit_function_std is not None:
                y_fit_std = fit_function_std(x_fit, *pstd)
                ax.fill_between(
                    x_fit,
                    y_fit + y_fit_std,
                    y_fit - y_fit_std,
                    alpha=0.3,
                    color=color_dict[state_name],
                )
        ax.scatter(
            x[filter],
            y[filter],
            # c=x[filter],
            # cmap=mpl.cm.viridis,
            color=color_dict[state_name],
            alpha=0.7,
            marker=marker_dict.get(state_name, "."),
            # marker="$f=ma$",
            s=20,
            # s=2000,
            label=f"{state_name} state",
            zorder=1,
        )
    if legend:
        ax.legend()
    ax.set_xlabel("N")
    ax.set_ylabel(label_dict[metric])
    return fig


#################################################################################
# compute sign changes and add to data
for data in [data_excited, data_superradient]:
    for res in data:
        res["st_sign_change"] = list()
        for st in res["st"]:
            sign_slope = np.sign(st)
            sign_change = np.sign(sign_slope[:-1] - sign_slope[1:])
            res["st_sign_change"].append(sign_change)

#################################################################################
# set figure size
fig_width = 0.8 * latex_textwidth
fig_height = 0.25 * latex_textheight

#################################################################################
# average after initial phase (fig. 5b)

for data in [data_excited, data_superradient]:
    for res in data:
        res["avg after first bump"] = list()
        for et, st_sign_change in zip(res["et"], res["st_sign_change"]):
            # get first non zero
            first_sign_change = (st_sign_change != 0).argmax(axis=0)
            after_first_bump = et[first_sign_change + 1 :]
            res["avg after first bump"].append(np.average(after_first_bump))


for data in [data_excited, data_superradient]:
    for res in data:
        res["avg after first bump over N"] = list()
        for avg in res["avg after first bump"]:
            # get first non zero
            N = res["N"]
            res["avg after first bump over N"].append(avg / N)

fig_avg_after_first_bump = plot_data_N_dependence(
    [data_superradient, data_excited],
    "avg after first bump",
    # max_N=43,
    fit_function=quadratic_linear,
    fit_function_std=quadratic_linear_std,
    initial_fit_parameters=(1, 0),
    width=fig_width,
)

fig_avg_after_first_bump.set_figheight(fig_height)

fig_avg_after_first_bump.savefig(
    "figures/fig_avg_after_first_bump.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
fig_avg_after_first_bump.savefig(
    "figures/fig_avg_after_first_bump.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)
plt.close()

#################################################################################
# Max in intial phase (fig. 5a)
for data in [data_excited, data_superradient]:
    for res in data:
        res["first high"] = list()
        for et, st_sign_change in zip(res["et"], res["st_sign_change"]):
            # get first non zero
            first_sign_change = (st_sign_change != 0).argmax(axis=0)
            first_bump = et[: first_sign_change + 1]
            max_in_first_bump = np.max(first_bump)

            res["first high"].append(max_in_first_bump)

for data in [data_excited, data_superradient]:
    for res in data:
        res["first high over N"] = list()
        for a in res["first high"]:
            # get first non zero
            N = res["N"]
            res["first high over N"].append(a / N)

fig_high_of_first_bump = plot_data_N_dependence(
    [data_excited, data_superradient],
    "first high",
    # max_N=43,
    fit_function=quadratic_linear,
    fit_function_std=quadratic_linear_std,
    initial_fit_parameters=(1, 0),
    width=fig_width,
    legend=True,
)
fig_high_of_first_bump.set_figheight(fig_height)

fig_high_of_first_bump.savefig(
    "figures/fig_high_of_first_bump.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
fig_high_of_first_bump.savefig(
    "figures/fig_high_of_first_bump.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)
plt.close()

#################################################################################
# initial gradient (fig. 5c)
for data in [
    data_superradient,
    data_excited,
]:
    for res in data:
        res["initial slope"] = list()
        for et in res["et"]:
            # get first non zero
            # first_sign_change = (st_sign_change != 0).argmax(axis=0)
            # selection = et[: np.int32(np.round(first_sign_change / 10))]
            selection = et[: (np.array(res["tt"]) > 0.1).argmax(axis=0)]
            max_in_selection = np.max(selection)
            t_of_max_in_selection = res["tt"][np.argmax(selection)]
            res["initial slope"].append(
                max_in_selection / t_of_max_in_selection
            )

for data in [data_excited, data_superradient]:
    for res in data:
        res["initial slope over N"] = list()
        for a in res["initial slope"]:
            # get first non zero
            N = res["N"]
            res["initial slope over N"].append(a / N)

fig_slope_short_time = plot_data_N_dependence(
    [data_superradient, data_excited],
    "initial slope",
    # max_N=43,
    fit_function=quadratic_linear,
    fit_function_std=quadratic_linear_std,
    initial_fit_parameters=(1, 0),
    width=fig_width,
)
fig_slope_short_time.set_figheight(fig_height)
fig_slope_short_time.savefig(
    "figures/fig_slope_short_time.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
fig_slope_short_time.savefig(
    "figures/fig_slope_short_time.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)
plt.close()


#################################################################################
# create the demo plot fig. 4
demonstration_selection = [
    50,
]

data = data_superradient

fig_demo = plot_data(
    data,
    # max_N=43,
    N_selection=demonstration_selection,
    # x=(0, 6),
    y=(0, 65),
    width=latex_textwidth,
)

slopes_demos = plot_data(
    data,
    # x=(0, 6),
    # max_N=43,
    N_selection=demonstration_selection,
    metric="slopes",
    width=latex_textwidth,
)

slopes_demos.set_figheight(0.3 * latex_textwidth)
fig_demo.set_figheight(0.3 * latex_textwidth)

for i, N in enumerate(demonstration_selection):
    res_demo = data[-(N - 1)]

    assert res_demo["N"] == N, f"{res_demo['N']},  {N}"

    sign_change = res_demo["st_sign_change"][0]

    axes = fig_demo.get_axes()
    ax = axes[0]

    first_sign_change = (sign_change != 0).argmax(axis=0)
    after_first_bump = res_demo["et"][0][first_sign_change:]
    t_after_first_bump = res_demo["tt"][first_sign_change:]

    # filter = sign_change != 0

    lines = ax.get_lines()
    line = lines[i]
    line.set_alpha(0.2)
    color = line.get_color()

    # plot after first bump
    ax.plot(
        t_after_first_bump,
        np.real_if_close(after_first_bump),
        "o",
        color=color,
        markersize=line.get_markersize(),
        markeredgewidth=line.get_markeredgewidth(),
    )
    # plot average line
    ax.hlines(
        y=np.real_if_close(res_demo["avg after first bump"][0]),
        xmin=min(t_after_first_bump),
        xmax=max(t_after_first_bump),
        linestyles="dashed",
        linewidths=2,
        alpha=0.7,
        colors="tab:red",
        label=f"Average after initial slope for N={N}",
    )

    # mark boundary of first bump
    ax.vlines(
        x=np.array(res_demo["tt"][:-2])[first_sign_change],
        ymin=0,
        ymax=np.real_if_close(
            np.array(res_demo["et"][0])[:-2][first_sign_change]
        ),
        colors="tab:red",
        alpha=0.7,
    )

    # mark value of first sign change
    ax.plot(
        # [:-2] to match the length of the boolean array: the slopes have one value less, and the sign changes one more less entry
        np.array(res_demo["tt"][:-2])[first_sign_change],
        np.real_if_close(np.array(res_demo["et"][0])[:-2][first_sign_change]),
        "o",
        markersize=3.5,
        markeredgewidth=1.5,
        markerfacecolor="none",
        color="tab:red",
        label="First sign change",
    )

    # add position of first sign change to  ticks
    # ticks = ax.get_xticks()
    # tick_labels = ax.get_xticklabels()
    # ax.set_xticks(
    #     np.append(ticks, np.array(res_demo["tt"][:-2])[first_sign_change])
    # )
    # ax.set_xticklabels(np.append(tick_labels, ""))

    # steepness of 0 to 0.1
    xtriangle = np.linspace(0, 0.1, 100)
    ytriangle = np.real(res_demo["initial slope"][0] * xtriangle)
    ax.plot(
        xtriangle, ytriangle, color="tab:red", label="Avg. slope for $t=0.1$"
    )
    # ax.legend()
    # # inset Axes....
    # x1, x2, y1, y2 = (
    #     0,
    #     0.15,
    #     0,
    #     np.max(ytriangle) * 1.2,
    # )  # subregion of the original image
    # axins = ax.inset_axes(
    #     [0.64, 0.03, 0.33, 0.33],
    #     xlim=(x1, x2),
    #     ylim=(y1, y2),
    #     xticklabels=[],
    #     yticklabels=[],
    # )
    # axins.plot(xtriangle, ytriangle, color="tab:red")
    # ax.indicate_inset_zoom(axins, edgecolor="black")

    # edit the slopes plot
    axes_slopes = slopes_demos.get_axes()
    ax_slopes = axes_slopes[0]
    ax_slopes.plot(
        np.array(res_demo["tt"][:-2])[first_sign_change],
        np.real_if_close(np.array(res_demo["st"][0])[:-1][first_sign_change]),
        "o",
        markersize=3.5,
        markeredgewidth=1.5,
        markerfacecolor="none",
        color="r",
        label="first sign change",
    )

    frame_ymin, frame_ymax = ax_slopes.get_ylim()
    # y_length = ymax + abs(ymin)
    # yvalue = np.real_if_close(
    #     np.array(res_demo["st"][0])[:-2][first_sign_change]
    # )

    # ax_slopes.axvline(
    #     x=np.array(res_demo["tt"][:-2])[first_sign_change],
    #     ymin=0,
    #     ymax=yvalue / y_length,
    #     color=color,
    #     alpha=0.7,
    # )

    ax_slopes.vlines(
        x=np.array(res_demo["tt"][:-2])[first_sign_change],
        ymin=frame_ymin,
        ymax=np.real_if_close(
            np.array(res_demo["st"][0])[:-2][first_sign_change]
        ),
        colors="tab:red",
        alpha=0.7,
    )
    ax_slopes.set_ylim(frame_ymin, frame_ymax)
    # ax_slopes.legend()

fig_demo.savefig(
    "figures/fig_demo.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
fig_demo.savefig(
    "figures/fig_demo.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)

slopes_demos.savefig(
    "figures/fig_slopes_demo.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)
slopes_demos.savefig(
    "figures/fig_slopes_demo.png",
    bbox_inches="tight",
    format="png",
    dpi=500,
    transparent=True,
)
plt.close()


#################################################################################
# plot the old data set "..._long.pickles" for demonstrating anomalies (fig. 1)
fig_anomalies = plot_data(
    data_excited_long,
    # max_N=43,
    x=(0, 3),
    y=(0, 100),
    width=latex_textwidth * 0.5,
    N_selection=[50],
    max_N_color=60,
    # use_larger_font=True,
)

fig_anomalies.savefig(
    "figures/fig_anomalies.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)

fig_no_anomalies = plot_data(
    data_excited,
    # max_N=43,
    x=(0, 3),
    y=(0, 100),
    width=latex_textwidth * 0.5,
    N_selection=[50],
    max_N_color=60,
    # use_larger_font=True,
)

fig_no_anomalies.savefig(
    "figures/fig_no_anomalies.pdf",
    bbox_inches="tight",
    format="pdf",
    dpi=500,
    transparent=True,
)

plt.close()
