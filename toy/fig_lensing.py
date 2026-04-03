"""
NeurIPS-style lensing figure.

Layout (2 rows × 3 columns):
  Row 0: (a) Uncontrolled  (b) Concave Lens   (c) Convex Lens
  Row 1: (d) Flat Profile   (e) Concave Loss   (f) Convex Loss

Each trajectory panel shows:
  - p_data (red, source distribution at t=0)
  - p_ref  (blue, particles at t=T)
  - trajectories (green lines connecting them)

Usage:
    python -m toy.fig_lensing
"""

import pickle, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax

from toy.distributions import DISTRIBUTIONS, nu_lensing, nu_convex, nu_flat
from toy.model import build_w_net
from toy.train import build_train_fns

# ── Config ───────────────────────────────────────────────────────────────────

VARIANTS = [
    ('Uncontrolled',  'toy/outputs/lensing_uncontrolled', None),
    ('Concave Lens',  'toy/outputs/lensing_concave',      nu_lensing),
    ('Convex Lens',   'toy/outputs/lensing_convex',       nu_convex),
    ('Flat Profile',  'toy/outputs/lensing_flat',         nu_flat),
]

LOSS_PANELS = [
    ('Concave Lens Loss', 'toy/outputs/lensing_concave'),
    ('Convex Lens Loss',  'toy/outputs/lensing_convex'),
]

LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

COLOR_DATA = '#e74c3c'   # red — source p_data
COLOR_REF  = '#3498db'   # blue — target p_ref
COLOR_TRAJ = '#2ecc71'   # green — trajectories
COLOR_LOSS = '#2c7bb6'
SMOOTH_WIN = 12

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_run(path):
    with open(f'{path}/w_params.pkl', 'rb') as f:
        w_params = pickle.load(f)
    with open(f'{path}/loss_lst.pkl', 'rb') as f:
        loss_lst = pickle.load(f)
    with open(f'{path}/config_used.yaml') as f:
        cfg = yaml.safe_load(f)
    # Load saved trajectory if available
    traj = None
    try:
        with open(f'{path}/traj_focusing.pkl', 'rb') as f:
            traj = pickle.load(f)
    except FileNotFoundError:
        pass
    return w_params, loss_lst, cfg, traj


def smooth(arr, w=SMOOTH_WIN):
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def get_focusing_traj(w_params, cfg, nu_fn, key):
    """Generate a focusing trajectory (source → target)."""
    N = cfg['N']
    init_sample = DISTRIBUTIONS[cfg['distribution']]
    target_pos = jnp.array([cfg['target_pos']])

    w_net = build_w_net(cfg)
    fns = build_train_fns(w_net, optax.adam(cfg['lr']), cfg, nu_fn=nu_fn)
    rollout = fns['rollout']

    key, k1, k2 = jax.random.split(key, 3)
    pos0 = init_sample(k1, N)
    _, _, traj, _, _ = rollout(pos0, k2, w_params, None,
                               "focusing", "pretraining", target_pos, 0)
    return np.array(traj), key


def plot_trajectories(ax, traj, label, panel_label):
    """Plot source, target, and trajectory lines."""
    n_steps, N, _ = traj.shape
    pos_start = traj[0]    # p_data (source)
    pos_end   = traj[-1]   # p_ref (target)

    # Trajectory lines (subsample particles for clarity)
    n_show = min(N, 150)
    for i in range(n_show):
        ax.plot(traj[:, i, 0], traj[:, i, 1],
                color=COLOR_TRAJ, alpha=0.15, linewidth=0.3, rasterized=True)

    # Source and target scatter
    ax.scatter(pos_start[:, 0], pos_start[:, 1],
               s=3, color=COLOR_DATA, alpha=0.6, linewidths=0, zorder=3,
               rasterized=True, label=r'$p_{\mathrm{data}}$')
    ax.scatter(pos_end[:, 0], pos_end[:, 1],
               s=3, color=COLOR_REF, alpha=0.6, linewidths=0, zorder=3,
               rasterized=True, label=r'$p_{\mathrm{ref}}$')

    ax.set_title(f'{panel_label}  {label}', fontsize=9, pad=4, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('black')


# ── Figure ────────────────────────────────────────────────────────────────────

def build():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 8,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.0))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.35, wspace=0.20,
                        left=0.05, right=0.98, top=0.92, bottom=0.10)

    key = jax.random.PRNGKey(42)

    # Collect all trajectory extents for shared axis limits
    all_trajs = []

    # ── Top row: Uncontrolled, Concave, Convex ────────────────────────────
    top_variants = VARIANTS[:3]
    for col, (label, path, nu_fn) in enumerate(top_variants):
        w_params, loss_lst, cfg, traj = load_run(path)
        if traj is None:
            traj, key = get_focusing_traj(w_params, cfg, nu_fn, key)
        all_trajs.append(traj)
        plot_trajectories(axes[0, col], traj, label, LABELS[col])

    # ── Bottom-left: Flat Profile ─────────────────────────────────────────
    label, path, nu_fn = VARIANTS[3]
    w_params, loss_lst, cfg, traj = load_run(path)
    if traj is None:
        traj, key = get_focusing_traj(w_params, cfg, nu_fn, key)
    all_trajs.append(traj)
    plot_trajectories(axes[1, 0], traj, label, LABELS[3])

    # ── Shared axis limits across all trajectory panels ───────────────────
    all_pts = np.concatenate([t.reshape(-1, 2) for t in all_trajs], axis=0)
    x_min, x_max = np.percentile(all_pts[:, 0], [1, 99])
    y_min, y_max = np.percentile(all_pts[:, 1], [1, 99])
    pad = 0.15 * max(x_max - x_min, y_max - y_min)
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]:
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    # Legend on first panel
    axes[0, 0].legend(fontsize=6, loc='lower left', framealpha=0.7,
                      markerscale=2, handletextpad=0.3)

    # ── Bottom-center and bottom-right: Loss curves ───────────────────────
    for col_offset, (label, path) in enumerate(LOSS_PANELS):
        col = col_offset + 1
        _, loss_lst, cfg, _ = load_run(path)
        ax = axes[1, col]
        raw = np.array(loss_lst)
        sm = smooth(raw)
        ax.plot(raw, color=COLOR_LOSS, alpha=0.15, linewidth=0.4)
        ax.plot(np.arange(len(sm)) + SMOOTH_WIN // 2, sm,
                color=COLOR_LOSS, linewidth=1.0)
        ax.set_xlim(0, len(raw))
        y_max = float(np.max(sm)) * 1.15
        ax.set_ylim(0, y_max)
        ax.set_xlabel('Epoch', fontsize=8, labelpad=2)
        ax.set_ylabel(r'$\mathcal{L}_{\mathrm{total}}$', fontsize=9, labelpad=2)
        ax.set_title(f'{LABELS[4 + col_offset]}  {label}', fontsize=9, pad=4, loc='left')
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.grid(True, which='major', linewidth=0.4, alpha=0.25, color='#888888')
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('black')
        ax.tick_params(labelsize=7, width=0.5, colors='black')

    return fig


if __name__ == '__main__':
    fig = build()
    fig.savefig('toy/outputs/fig_lensing.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('toy/outputs/fig_lensing.png', dpi=200, bbox_inches='tight')
    print('Saved toy/outputs/fig_lensing.pdf / .png')
