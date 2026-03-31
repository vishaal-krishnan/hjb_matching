"""
Generate all figures from a saved experiment run.

Usage:
    python -m toy.make_figures --outputs toy/outputs/4gaussians --out_dir toy/figures/4gaussians
"""

import argparse
import pickle
import yaml
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import jax
import jax.numpy as jnp
import optax

from toy.distributions import DISTRIBUTIONS, nu_lensing
from toy.model import build_w_net
from toy.train import build_train_fns


def load_run(outputs_dir):
    with open(f'{outputs_dir}/w_params.pkl', 'rb') as f:
        w_params = pickle.load(f)
    with open(f'{outputs_dir}/loss_lst.pkl', 'rb') as f:
        loss_lst = pickle.load(f)
    with open(f'{outputs_dir}/config_used.yaml') as f:
        cfg = yaml.safe_load(f)
    return w_params, loss_lst, cfg


def fig_loss_curve(loss_lst, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_lst, color='steelblue', linewidth=1.0)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/loss_curve.png', dpi=200)
    plt.close()
    print('  saved loss_curve.png')


def fig_value_function(w_net, w_params, cfg, out_dir):
    L = cfg['L']
    steps = cfg['steps']
    resolution = 150

    x = jnp.linspace(-L, L, resolution)
    y = jnp.linspace(-L, L, resolution)
    xx, yy = jnp.meshgrid(x, y)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    time_indices = [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]
    t_labels = [f't={i/(steps-1):.2f}' for i in time_indices]

    fig, axs = plt.subplots(1, len(time_indices), figsize=(4 * len(time_indices), 4), sharey=True)

    W_all = [w_net.apply(w_params, grid, jnp.full((grid.shape[0],), s)).squeeze().reshape(resolution, resolution)
             for s in time_indices]
    vmin = min(w.min() for w in W_all)
    vmax = max(w.max() for w in W_all)

    for i, (W, label) in enumerate(zip(W_all, t_labels)):
        pc = axs[i].pcolormesh(xx, yy, W, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i].set_title(label, fontsize=12)
        axs[i].set_aspect('equal')
        axs[i].set_xlim(-L, L)
        axs[i].set_ylim(-L, L)

    fig.colorbar(pc, ax=axs[-1], fraction=0.046, pad=0.04)
    plt.suptitle('Value function W(x, t)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/value_function.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved value_function.png')


def fig_combined_snapshots(w_net, rollout, w_params, cfg, key, out_dir):
    L = cfg['L']
    steps = cfg['steps']
    N = cfg['N']
    target_pos = jnp.array([cfg['target_pos']])
    reverse_init_std = cfg.get('reverse_init_std', 0.25)
    resolution = 150

    s_indices_W = [steps - 1, steps // 2, 1]
    s_indices_p = [0, steps // 2, steps - 1]
    norm_times = [0.0, 0.5, 1.0]

    x = jnp.linspace(-L, L, resolution)
    y = jnp.linspace(-L, L, resolution)
    xx, yy = jnp.meshgrid(x, y)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    W_snapshots = [
        w_net.apply(w_params, grid, jnp.full((grid.shape[0],), s)).squeeze().reshape(resolution, resolution)
        for s in s_indices_W
    ]

    key, k1, k2 = jax.random.split(key, 3)
    pos0_rev = jax.random.normal(k1, (N, 2)) * reverse_init_std + target_pos
    _, _, trajectory, _, _ = rollout(pos0_rev, k2, w_params, None, "reversing", "pretraining", target_pos, 0)

    fig, axs = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=True)
    vmin = min(w.min() for w in W_snapshots)
    vmax = max(w.max() for w in W_snapshots)

    for i, (W, s_idx, t_norm) in enumerate(zip(W_snapshots, s_indices_W, norm_times)):
        pc = axs[i, 0].pcolormesh(xx, yy, W, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f'W(x, t={t_norm:.1f})', fontsize=12)
        axs[i, 0].set_aspect('equal')

        pos = trajectory[s_indices_p[i]]
        axs[i, 1].scatter(pos[:, 0], pos[:, 1], color='red', s=1, alpha=0.6)
        axs[i, 1].set_title(f'Particles at t={t_norm:.1f}', fontsize=12)
        axs[i, 1].set_xlim(-L, L)
        axs[i, 1].set_ylim(-L, L)
        axs[i, 1].set_aspect('equal')

    ax_cb = inset_axes(axs[2, 0], width="5%", height="60%", loc="lower left", borderpad=2)
    cbar = fig.colorbar(pc, cax=ax_cb, orientation='vertical')
    cbar.ax.tick_params(labelsize=9, colors='white')

    plt.subplots_adjust(wspace=0.05, hspace=0.12, left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.savefig(f'{out_dir}/combined_snapshots.png', dpi=200)
    plt.close()
    print('  saved combined_snapshots.png')


def fig_particle_animation(rollout, init_sample, w_params, cfg, key, step_mode, out_dir, nu_fn=None):
    L = cfg['L']
    N = cfg['N']
    target_pos = jnp.array([cfg['target_pos']])
    reverse_init_std = cfg.get('reverse_init_std', 0.25)

    key, k1, k2 = jax.random.split(key, 3)
    if step_mode == 'focusing':
        pos0 = init_sample(k1, N)
    else:
        pos0 = jax.random.normal(k1, (N, 2)) * reverse_init_std + target_pos

    _, _, traj, _, _ = rollout(pos0, k2, w_params, None, step_mode, "pretraining", target_pos, 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], color='red', s=1, alpha=0.6)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect('equal')
    ax.set_title(step_mode, fontsize=12)

    def update(frame):
        scat.set_offsets(traj[frame])
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=traj.shape[0], interval=40, blit=True)
    fname = f'{out_dir}/animation_{step_mode}.gif'
    ani.save(fname, writer='pillow', fps=25, dpi=120)
    plt.close()
    print(f'  saved animation_{step_mode}.gif')
    return key


def fig_w_field_animation(w_net, w_params, cfg, out_dir):
    L = cfg['L']
    steps = cfg['steps']
    resolution = 100

    x = jnp.linspace(-L, L, resolution)
    y = jnp.linspace(-L, L, resolution)
    xx, yy = jnp.meshgrid(x, y)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    stride = max(1, steps // 40)
    s_values = list(range(0, steps, stride))
    W_frames = [
        w_net.apply(w_params, grid, jnp.full((grid.shape[0],), s)).squeeze().reshape(resolution, resolution)
        for s in s_values
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    pc = ax.pcolormesh(xx, yy, W_frames[0], shading='auto', cmap='viridis')
    plt.colorbar(pc, ax=ax)
    title = ax.set_title(f'W(x, t=0.00)')
    ax.set_aspect('equal')

    def update(frame):
        pc.set_array(W_frames[frame].ravel())
        t_norm = s_values[frame] / (steps - 1)
        title.set_text(f'W(x, t={t_norm:.2f})')
        return pc,

    ani = animation.FuncAnimation(fig, update, frames=len(s_values), interval=100)
    ani.save(f'{out_dir}/w_field_animation.gif', writer='pillow', fps=15, dpi=120)
    plt.close()
    print('  saved w_field_animation.gif')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', required=True, help='Path to saved outputs directory')
    parser.add_argument('--out_dir', default=None, help='Where to save figures (default: <outputs>/figures/)')
    args = parser.parse_args()

    out_dir = out_dir if out_dir else os.path.join(args.outputs, 'figures')
    out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading from {args.outputs}...')
    print(f'Saving figures to {out_dir}/')
    w_params, loss_lst, cfg = load_run(args.outputs)

    init_sample = DISTRIBUTIONS[cfg['distribution']]
    nu_fn = nu_lensing if cfg.get('use_analytical_nu', False) else None
    target_pos = jnp.array([cfg['target_pos']])

    w_net = build_w_net(cfg)
    w_opt = optax.adam(cfg['lr'])
    w_opt_state = w_opt.init(w_params)
    fns = build_train_fns(w_net, w_opt, cfg, nu_fn=nu_fn)
    rollout = fns['rollout']

    key = jax.random.PRNGKey(42)

    print('Generating figures...')
    fig_loss_curve(loss_lst, out_dir)
    fig_value_function(w_net, w_params, cfg, out_dir)
    fig_combined_snapshots(w_net, rollout, w_params, cfg, key, out_dir)
    key = fig_particle_animation(rollout, init_sample, w_params, cfg, key, 'focusing', out_dir, nu_fn)
    key = fig_particle_animation(rollout, init_sample, w_params, cfg, key, 'reversing', out_dir, nu_fn)
    fig_w_field_animation(w_net, w_params, cfg, out_dir)

    print(f'\nAll figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
