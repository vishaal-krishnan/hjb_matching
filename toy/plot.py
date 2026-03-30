import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_training_progress(epoch, loss_lst, final_pos, pos0, target_pos, w_net, w_params, cfg, nu_fn=None):
    L = cfg['L']
    resolution = 100

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    if nu_fn is not None:
        x_grid = jnp.linspace(-5, 5, resolution)
        y_grid = jnp.linspace(-5, 5, resolution)
        x_mesh, y_mesh = jnp.meshgrid(x_grid, y_grid)
        pos_mesh = jnp.stack([x_mesh, y_mesh], axis=-1).reshape(-1, 2)
        nu_mesh = nu_fn(pos_mesh).reshape(x_mesh.shape)
        cf = axs[0].contourf(x_mesh, y_mesh, nu_mesh, levels=100)
        plt.colorbar(cf, ax=axs[0])

    axs[0].scatter(final_pos[:, 0], final_pos[:, 1], s=1, c='red', label='Final')
    axs[0].scatter(target_pos[:, 0], target_pos[:, 1], s=10, c='black', label='Target')
    axs[0].scatter(pos0[:, 0], pos0[:, 1], s=1, c='blue', label='Initial')
    axs[0].set_title("Final Configuration")
    axs[0].legend()
    axs[0].set_xlim(-L, L)
    axs[0].set_ylim(-L, L)
    axs[0].set_aspect('equal')

    loss_arr = jnp.array(loss_lst)
    axs[1].plot(loss_arr / jnp.max(loss_arr))
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss (normalized)")
    axs[1].set_title("Training Loss")

    x = jnp.linspace(-L, L, resolution)
    y = jnp.linspace(-L, L, resolution)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    s_array = jnp.full((grid_points.shape[0],), 0)
    w = w_net.apply(w_params, grid_points, s_array).reshape(resolution, resolution)
    pc = axs[2].pcolormesh(xx, yy, w, shading='auto', cmap='viridis')
    plt.colorbar(pc, ax=axs[2])
    axs[2].set_title("Value function w(x, s=0)")
    axs[2].set_xlim(-L, L)
    axs[2].set_ylim(-L, L)
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.show()


def animate_w_field(w_net, w_params, cfg, step_stride=1):
    L = cfg['L']
    steps = cfg['steps']
    resolution = 100

    x = jnp.linspace(-L, L, resolution)
    y = jnp.linspace(-L, L, resolution)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    s_values = jnp.arange(0, steps, step_stride)
    W_frames = []
    for s in s_values:
        s_array = jnp.full((grid_points.shape[0],), s)
        W_vals = w_net.apply(w_params, grid_points, s_array).squeeze()
        W_frames.append(W_vals.reshape(resolution, resolution))

    fig, ax = plt.subplots(figsize=(4, 4))
    pc = ax.pcolormesh(xx, yy, W_frames[0], shading='auto', cmap='viridis')
    plt.colorbar(pc, ax=ax)
    title = ax.set_title(f"W(x, step={int(s_values[0])})")
    ax.set_aspect('equal')

    def update(frame):
        pc.set_array(W_frames[frame].ravel())
        title.set_text(f"W(x, step={int(s_values[frame])})")
        return pc,

    ani = animation.FuncAnimation(fig, update, frames=len(s_values), interval=500)
    ani.save('W_field.gif', writer='pillow', fps=25, dpi=150)
    plt.close()
    return ani


def make_video(rollout_fn, init_sample_fn, w_params, cfg, key1, key2,
               step_mode="focusing", steps_beyond_horizon=0, nu_fn=None):
    L = cfg['L']
    N = cfg['N']
    steps = cfg['steps']
    T_tot = cfg['T_tot']
    guidance_strength = cfg['guidance_strength']
    reverse_init_std = cfg.get('reverse_init_std', 0.25)
    target_pos = jnp.array([cfg['target_pos']])

    if step_mode == "focusing":
        pos0_viz = init_sample_fn(key1, N)
    elif step_mode == "reversing":
        pos0_viz = jax.random.normal(key1, (N, 2)) * reverse_init_std + target_pos
    else:
        raise ValueError(f"Invalid step_mode: {step_mode}")

    _, _, trajectory, _, _ = rollout_fn(
        pos0_viz, key2, w_params, None, step_mode, "pretraining",
        target_pos, steps_beyond_horizon
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    scat = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], color='red', s=1)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect('equal')
    ax.set_title(step_mode)

    def update(frame):
        scat.set_offsets(trajectory[frame])
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=trajectory.shape[0] - 1, interval=40, blit=True)
    ani.save(f'{step_mode}.gif', writer='pillow', fps=25, dpi=150)

    if nu_fn is not None:
        _plot_static_trajectories(trajectory, step_mode, cfg, nu_fn)

    plt.show()
    return ani


def _plot_static_trajectories(trajectory, step_mode, cfg, nu_fn):
    L = cfg['L']
    N = trajectory.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))

    x_grid = jnp.linspace(-5, 5, 100)
    y_grid = jnp.linspace(-5, 5, 100)
    x_mesh, y_mesh = jnp.meshgrid(x_grid, y_grid)
    pos_mesh = jnp.stack([x_mesh, y_mesh], axis=-1).reshape(-1, 2)
    nu_mesh = nu_fn(pos_mesh).reshape(x_mesh.shape)
    cf = ax.contourf(x_mesh, y_mesh, nu_mesh, levels=100, cmap='gray')
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label(r'$\nu$', fontsize=20)
    ax.axis('off')

    for i in range(0, N, 10):
        traj_i = trajectory[:, i, :]
        ax.plot(traj_i[:, 0], traj_i[:, 1], linewidth=0.5, alpha=0.8, color='lightgreen')

    if step_mode == "focusing":
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], color='salmon', s=3, label='Initial', zorder=5)
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], color='skyblue', s=3, label='Final', zorder=5)
    elif step_mode == "reversing":
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], color='salmon', s=3, label='Initial', zorder=5)
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], color='skyblue', s=3, label='Final', zorder=5)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect('equal')
    fig.savefig(f"trajectories_{step_mode}.pdf", bbox_inches='tight')
    plt.show()


def save_combined_snapshots(w_net, rollout_fn, w_params, cfg, key1, key2):
    L = cfg['L']
    steps = cfg['steps']
    N = cfg['N']
    target_pos = jnp.array([cfg['target_pos']])
    reverse_init_std = cfg.get('reverse_init_std', 0.25)
    resolution = 100

    s_indices_W = [steps - 1, steps // 2, 1]
    s_indices_p = [0, steps // 2, steps - 1]
    norm_times = [0.0, 0.5, 1.0]

    x = jnp.linspace(-L, L, resolution)
    y = jnp.linspace(-L, L, resolution)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    W_snapshots = []
    for s in s_indices_W:
        s_array = jnp.full((grid_points.shape[0],), s)
        W_vals = w_net.apply(w_params, grid_points, s_array).squeeze()
        W_snapshots.append(W_vals.reshape(resolution, resolution))

    pos0_viz = jax.random.normal(key1, (N, 2)) * reverse_init_std + target_pos
    _, _, trajectory, _, _ = rollout_fn(
        pos0_viz, key2, w_params, None, "reversing", "pretraining", target_pos, 0
    )

    fig, axs = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=True)
    vmin = min(w.min() for w in W_snapshots)
    vmax = max(w.max() for w in W_snapshots)

    for i, (W, s_idx, t_norm) in enumerate(zip(W_snapshots, s_indices_W, norm_times)):
        pc = axs[i, 0].pcolormesh(xx, yy, W, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f"W(x, t={t_norm:.2f})", fontsize=12, pad=6)
        axs[i, 0].set_aspect('equal')

        pos = trajectory[s_indices_p[i]]
        axs[i, 1].scatter(pos[:, 0], pos[:, 1], color='red', s=1)
        axs[i, 1].set_title(f"Particles at t={t_norm:.2f}", fontsize=12, pad=6)
        axs[i, 1].set_xlim(-L, L)
        axs[i, 1].set_ylim(-L, L)
        axs[i, 1].set_aspect('equal')

    ax_cb = inset_axes(axs[2, 0], width="5%", height="60%", loc="lower left", borderpad=2)
    cbar = fig.colorbar(pc, cax=ax_cb, orientation='vertical')
    cbar.ax.tick_params(labelsize=10, colors='white')

    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig("combined_snapshots_3x2.png", dpi=300)
    plt.show()
    plt.close()
