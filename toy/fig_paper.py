"""
NeurIPS-style paper figure.

Each experiment is a column:
  [Loss curve  +  W(x,t) insets along top]
  [Generative process panels]

Usage:
    python -m toy.fig_paper
"""

import pickle, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    ('Swiss Roll',  'toy/outputs/swiss_roll'),
]

W_T_FRACS   = [0.1, 0.4, 0.7, 1.0]
GEN_T_FRACS = [0.0, 0.5, 1.0]

CMAP_W     = 'viridis'
COLOR_LOSS = '#2c7bb6'
COLOR_PART = '#d7191c'
SMOOTH_WIN = 12

# Inset geometry in axes-fraction coords
INSET_W = 0.20   # width  of each W inset
INSET_H = 0.30   # height of each W inset
INSET_Y = 0.66   # bottom of inset strip (top 34% of axes → above the curve)

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


def style_loss(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(labelsize=5.5, width=0.5)


def style_spatial(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('C')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color('#aaaaaa')


# ── Figure ────────────────────────────────────────────────────────────────────

def build():
    n_exp  = len(EXPERIMENTS)
    n_w    = len(W_T_FRACS)
    n_gen  = len(GEN_T_FRACS)

    fig = plt.figure(figsize=(3.5 * n_exp, 5.5))
    fig.patch.set_facecolor('white')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
    })

    # One column per experiment
    outer = gridspec.GridSpec(
        1, n_exp, figure=fig,
        wspace=0.32,
        left=0.09, right=0.97, top=0.93, bottom=0.07,
    )

    key = jax.random.PRNGKey(42)

    for col_idx, (label, path) in enumerate(EXPERIMENTS):
        w_params, loss_lst, cfg = load_run(path)
        L     = cfg['L']
        nu_fn = nu_lensing if cfg.get('use_analytical_nu', False) else None

        w_net   = build_w_net(cfg)
        fns     = build_train_fns(w_net, optax.adam(cfg['lr']), cfg, nu_fn=nu_fn)
        rollout = fns['rollout']

        # Each column: 2 rows — loss+W insets | gen panels
        col_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[col_idx],
            height_ratios=[2.2, 1.0],
            hspace=0.32,
        )

        # ── Loss curve ────────────────────────────────────────────────────
        ax_loss = fig.add_subplot(col_gs[0])
        raw = np.array(loss_lst)
        sm  = smooth(raw)
        ax_loss.plot(raw, color=COLOR_LOSS, alpha=0.18, linewidth=0.7)
        ax_loss.plot(np.arange(len(sm)) + SMOOTH_WIN // 2, sm,
                     color=COLOR_LOSS, linewidth=1.5)
        ax_loss.set_xlim(0, len(raw))
        ax_loss.set_ylim(bottom=0)
        ax_loss.set_xlabel('Epoch', fontsize=6.5, labelpad=2)
        ax_loss.set_ylabel(r'$\mathcal{L}_{\mathrm{total}}$', fontsize=8, labelpad=2)
        ax_loss.set_title(label, fontsize=9, fontweight='bold', pad=5)
        style_loss(ax_loss)

        # ── W insets: evenly spaced across top of loss panel ─────────────
        W_data = [w_field(w_net, w_params, tf, cfg) for tf in W_T_FRACS]
        vmin = min(W.min() for _, _, W in W_data)
        vmax = max(W.max() for _, _, W in W_data)

        # Space n_w insets evenly, leaving a small margin on each side
        margin = 0.02
        total_space = 1.0 - 2 * margin - n_w * INSET_W
        gap = total_space / (n_w - 1)

        for wi, ((xx, yy, W), tf) in enumerate(zip(W_data, W_T_FRACS)):
            x_c = margin + wi * (INSET_W + gap)
            ax_in = ax_loss.inset_axes([x_c, INSET_Y, INSET_W, INSET_H])
            ax_in.pcolormesh(xx, yy, W, cmap=CMAP_W, shading='nearest',
                             vmin=vmin, vmax=vmax, rasterized=True)
            style_spatial(ax_in)

            # t label just above each inset
            ax_loss.text(x_c + INSET_W / 2, INSET_Y + INSET_H + 0.02,
                         f'$t={tf:.2f}$',
                         transform=ax_loss.transAxes,
                         ha='center', va='bottom', fontsize=5.5, color='#333333')

        # Section label above the inset strip, centered
        ax_loss.text(0.5, INSET_Y + INSET_H + 0.10,
                     r'$W(x,\,t)$',
                     transform=ax_loss.transAxes,
                     ha='center', va='bottom', fontsize=7,
                     color='#444444', style='italic')

        # ── Generative process panels ─────────────────────────────────────
        gen_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_gen, subplot_spec=col_gs[1], wspace=0.06,
        )

        traj, key = generative_traj(rollout, w_params, cfg, key)
        n_frames  = traj.shape[0]
        frames    = [int(f * (n_frames - 1)) for f in GEN_T_FRACS]

        for gi, (fi, tf) in enumerate(zip(frames, GEN_T_FRACS)):
            ax = fig.add_subplot(gen_gs[gi])
            pts = traj[fi]
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=0.5, color=COLOR_PART, alpha=0.45, rasterized=True,
                       linewidths=0)
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            style_spatial(ax)
            ax.set_title(f'$t={tf:.1f}$', fontsize=6, pad=2)

            if gi == 0:
                ax.text(0.0, 1.18, 'Generative process',
                        transform=ax.transAxes,
                        fontsize=6.5, color='#444444', style='italic', va='bottom')

    return fig


if __name__ == '__main__':
    fig = build()
    fig.savefig('toy/outputs/fig_paper.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('toy/outputs/fig_paper.png', dpi=200, bbox_inches='tight')
    print('Saved toy/outputs/fig_paper.pdf / .png')
