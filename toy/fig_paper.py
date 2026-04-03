"""
NeurIPS-style paper figure.

3x3 grid: rows = Learned W, Generative process, Learning curve
          cols = experiments
Row labels on left (vertical), colorbar as vertical bar to right of W row.

Usage:
    python -m toy.fig_paper
"""

import pickle, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import jax
import jax.numpy as jnp
import optax

from toy.distributions import DISTRIBUTIONS, nu_lensing
from toy.model import build_w_net
from toy.train import build_train_fns

# ── Config ───────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ('4 Gaussians', 'toy/outputs/4gaussians'),
    ('Two Moons',   'toy/outputs/two_moon'),
    ('Swiss Roll',  'toy/outputs/swiss_roll'),
]

T_FRACS = [0.0, 0.33, 0.66, 1.0]

CMAP_W     = 'viridis_r'
COLOR_LOSS = '#2c7bb6'
COLOR_PART = '#d7191c'
SMOOTH_WIN = 12

LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)']

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_run(path):
    with open(f'{path}/w_params.pkl', 'rb') as f:
        w_params = pickle.load(f)
    with open(f'{path}/loss_lst.pkl', 'rb') as f:
        loss_lst = pickle.load(f)
    with open(f'{path}/config_used.yaml') as f:
        cfg = yaml.safe_load(f)
    return w_params, loss_lst, cfg


def smooth(arr, w=SMOOTH_WIN):
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def w_field(w_net, w_params, t_frac, cfg, res=90):
    L, steps = cfg['L'], cfg['steps']
    step_idx = max(1, int((1.0 - t_frac) * (steps - 2)) + 1)
    x = jnp.linspace(-L, L, res)
    y = jnp.linspace(-L, L, res)
    xx, yy = jnp.meshgrid(x, y)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    W = w_net.apply(w_params, grid, jnp.full((grid.shape[0],), step_idx)).squeeze().reshape(res, res)
    return np.array(xx), np.array(yy), np.array(W)


def generative_traj(rollout, w_params, cfg, key):
    N = cfg['N']
    target_pos = jnp.array([cfg['target_pos']])
    std = cfg.get('reverse_init_std', 0.25)
    key, k1, k2 = jax.random.split(key, 3)
    pos0 = jax.random.normal(k1, (N, 2)) * std + target_pos
    _, _, traj, _, _ = rollout(pos0, k2, w_params, None,
                               "reversing", "pretraining", target_pos, 0)
    return np.array(traj), key


def style_spatial(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color('#aaaaaa')


# ── Figure ────────────────────────────────────────────────────────────────────

def build():
    n_exp = len(EXPERIMENTS)
    n_t   = len(T_FRACS)

    fig_w = 6.75
    margin_l = 0.09   # room for vertical row labels
    margin_r = 0.02
    # Column geometry
    col_total = 1.0 - margin_l - margin_r
    col_span = col_total / n_exp
    panel_gap = 0.008  # gap between panels within a column

    # Compute panel size from column width
    col_w_in = fig_w * col_span
    pw_in = col_w_in / (n_t + (n_t - 1) * panel_gap / col_span)
    panel_h_in = pw_in  # square

    loss_h_in = 0.95
    title_h   = 0.22
    gap_w_gen = 0.03    # tiny gap between W and gen rows (inches)
    gap_gen_loss = 0.12  # gap between gen and loss rows
    bottom_h  = 0.35

    fig_h = title_h + 2 * panel_h_in + gap_w_gen + gap_gen_loss + loss_h_in + bottom_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

    def in2frac(inches):
        return inches / fig_h

    # Vertical positions (bottom-up)
    loss_bot = bottom_h / fig_h
    loss_top = loss_bot + in2frac(loss_h_in)

    gen_bot = loss_top + in2frac(gap_gen_loss)
    gen_top = gen_bot + in2frac(panel_h_in)

    w_bot = gen_top + in2frac(gap_w_gen)
    w_top = w_bot + in2frac(panel_h_in)

    title_y = w_top + in2frac(0.18)

    # ── Row labels (vertical text, left side) ────────────────────────────
    lbl_x = 0.005
    lbl_cx = margin_l / 2  # center of the label column
    # "Learned W" text + colorbar stacked tightly
    fig.text(lbl_cx, (w_bot + w_top) / 2 + 0.04, 'Learned $W$',
             fontsize=7.5, va='bottom', ha='center', color='black')
    fig.text(lbl_cx, (gen_bot + gen_top) / 2, 'Generative\nprocess',
             fontsize=7.5, va='center', ha='center', color='black',
             linespacing=1.3)
    # No row label for loss — the y-axis label serves that purpose

    key = jax.random.PRNGKey(42)
    global_vlo = np.inf
    global_vhi = -np.inf

    for col_idx, (label, path) in enumerate(EXPERIMENTS):
        w_params, loss_lst, cfg = load_run(path)
        L     = cfg['L']
        nu_fn = nu_lensing if cfg.get('use_analytical_nu', False) else None

        w_net   = build_w_net(cfg)
        fns     = build_train_fns(w_net, optax.adam(cfg['lr']), cfg, nu_fn=nu_fn)
        rollout = fns['rollout']

        col_left = margin_l + col_idx * col_span
        col_right = col_left + col_span - 0.025  # inter-column gap

        # ── Column title ─────────────────────────────────────────────────
        fig.text(col_left, title_y, f'{LABELS[col_idx]}  {label}',
                 fontsize=9.5, va='bottom', ha='left', color='black')

        # ── W panels ─────────────────────────────────────────────────────
        W_data = [w_field(w_net, w_params, tf, cfg) for tf in T_FRACS]
        _, _, W_last = W_data[-1]
        v_lo = float(np.percentile(W_last, 2))
        v_hi = float(np.percentile(W_last, 98))
        global_vlo = min(global_vlo, v_lo)
        global_vhi = max(global_vhi, v_hi)

        pw = (col_right - col_left - panel_gap * (n_t - 1)) / n_t
        ph = in2frac(panel_h_in)

        for wi, ((xx, yy, W), tf) in enumerate(zip(W_data, T_FRACS)):
            ax_l = col_left + wi * (pw + panel_gap)
            ax = fig.add_axes([ax_l, w_bot, pw, ph])
            ax.pcolormesh(xx, yy, W, cmap=CMAP_W, shading='nearest',
                          vmin=v_lo, vmax=v_hi, rasterized=True)
            style_spatial(ax)
            ax.set_title(f'$t={tf:.1f}$', fontsize=7, pad=2, color='black')

        # ── Gen panels (directly below W, same horizontal positions) ─────
        traj, key = generative_traj(rollout, w_params, cfg, key)
        n_frames = traj.shape[0]
        frames   = [int(f * (n_frames - 1)) for f in T_FRACS]

        for gi, (fi, tf) in enumerate(zip(frames, T_FRACS)):
            ax_l = col_left + gi * (pw + panel_gap)
            ax = fig.add_axes([ax_l, gen_bot, pw, ph])
            pts = traj[fi]
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=0.4, color=COLOR_PART, alpha=0.45, rasterized=True,
                       linewidths=0)
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            style_spatial(ax)

        # ── Loss curve (aligned with panels above) ─────────────────────
        ax_loss = fig.add_axes([col_left, loss_bot, col_right - col_left, in2frac(loss_h_in)])
        raw = np.array(loss_lst)
        sm  = smooth(raw)
        ax_loss.plot(raw, color=COLOR_LOSS, alpha=0.15, linewidth=0.4)
        ax_loss.plot(np.arange(len(sm)) + SMOOTH_WIN // 2, sm,
                     color=COLOR_LOSS, linewidth=1.0)
        ax_loss.set_xlim(0, len(raw))
        y_max = float(np.max(sm)) * 1.15
        ax_loss.set_ylim(0, y_max)
        ax_loss.set_xlabel('Epoch', fontsize=8, labelpad=2, color='black')
        ax_loss.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax_loss.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        ax_loss.grid(True, which='major', linewidth=0.4, alpha=0.25, color='#888888')

        if col_idx == 0:
            ax_loss.set_ylabel(r'Total Loss $\mathcal{L}$', fontsize=8, labelpad=2, color='black')
        else:
            ax_loss.set_yticklabels([])

        for sp in ax_loss.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('black')
        ax_loss.tick_params(labelsize=7, width=0.5, colors='black')

    # ── Horizontal colorbar under "Learned W" label in left margin ───────
    cbar_margin = 0.005
    cbar_left_pos = cbar_margin
    cbar_width = margin_l - 2 * cbar_margin
    cbar_h_frac = 0.015
    cbar_bot_pos = (w_bot + w_top) / 2 + 0.015
    cbar_ax = fig.add_axes([cbar_left_pos, cbar_bot_pos, cbar_width, cbar_h_frac])
    norm = mcolors.Normalize(vmin=global_vlo, vmax=global_vhi)
    sm_cbar = cm.ScalarMappable(cmap=CMAP_W, norm=norm)
    sm_cbar.set_array([])
    cbar = fig.colorbar(sm_cbar, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5, width=0.3, colors='black', pad=1)
    cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    cbar.outline.set_linewidth(0.3)

    return fig


if __name__ == '__main__':
    fig = build()
    fig.savefig('toy/outputs/fig_paper.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('toy/outputs/fig_paper.png', dpi=200, bbox_inches='tight')
    print('Saved toy/outputs/fig_paper.pdf / .png')
