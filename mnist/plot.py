import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import make_grid

from mnist.train import grad_w_fn, spatial_smoothness


def plot_loss_curves(loss_lst, loss_fk_lst, loss_dual_lst):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t = torch.tensor(loss_lst)
    axes[0].plot(t / t.max())
    axes[0].set_title('Total loss (normalized)')
    axes[1].plot(loss_fk_lst)
    axes[1].set_title('FK loss')
    axes[2].plot(loss_dual_lst)
    axes[2].set_title('Dual loss')
    plt.tight_layout()
    plt.show()


def plot_generated_samples(samples, n=16):
    samples = samples.clamp(-1.0, 1.0)
    samples = (samples + 1.0) / 2.0
    grid = make_grid(samples[:n], nrow=int(np.sqrt(n)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.title("Generated samples")
    plt.show()


def plot_w_trajectories(model, x, theta, D, T=1.0, dt=0.01, device='cuda'):
    model.eval()
    num_steps = int(T / dt)
    x_cur = x.clone()

    with torch.no_grad():
        w_0 = model(x_cur, torch.zeros(x_cur.shape[0], device=device))
        traj_w = [w_0.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(x_cur)
            drift = -theta * x_cur * dt
            diffusion = (2 * D * dt) ** 0.5 * noise
            x_cur = x_cur + drift + diffusion

            t_batched = torch.ones(x_cur.shape[0], device=device) * step * dt
            w = model(x_cur, t_batched).detach().cpu()
            traj_w.append(w)

    traj_w = torch.stack(traj_w, dim=1)
    traj_w = traj_w.view(traj_w.shape[0], traj_w.shape[1], -1)

    plt.figure(figsize=(10, 5))
    for i in range(traj_w.shape[0]):
        plt.plot(traj_w[i])
    plt.title('Trajectory of w (forward OU)')
    plt.xlabel('Forward time step')
    plt.show()


def plot_bump(model, x0, theta, D, T=1.0, dt=0.01, device='cuda'):
    model.eval()
    num_steps = int(T / dt)
    times = torch.linspace(0, T, num_steps + 1, device=x0.device)
    x = x0.clone()
    traj = [x0.clone()]

    for _ in range(num_steps):
        noise = torch.randn_like(x)
        x = x + (-theta * x * dt) + (2 * D * dt) ** 0.5 * noise
        traj.append(x.clone())

    traj = torch.stack(traj)
    traj = traj.permute(1, 0, *range(2, traj[0].ndim + 1))

    with torch.no_grad():
        for ind in [1, 10]:
            x_sel = traj[ind]
            for t in times[::11]:
                t_batched = torch.ones(x_sel.shape[0], device=device) * t
                w = model(x_sel, t_batched).detach().cpu()
                plt.plot(times.cpu(), w, label=f's={t:.2f}')
            plt.xlabel("Arc length", fontsize=14)
            plt.ylabel("w", fontsize=14)
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
            plt.grid(True)
            plt.show()


def create_generation_video(model, cfg, device='cuda'):
    theta = cfg['theta']
    D = cfg['D']
    num_steps = cfg['num_steps']
    batch_size = cfg.get('sample_batch_size', 16)

    time_steps = torch.linspace(1.0, 0.0, steps=num_steps + 2, device=device)[1:-1]
    step_size = time_steps[0] - time_steps[1]
    x = torch.randn(batch_size, 1, 28, 28, device=device) * (D / theta) ** 0.5
    frames = []

    for i, s in enumerate(time_steps):
        batch_t = torch.ones(batch_size, device=device) * s
        grad_w = grad_w_fn(model, x, batch_t)
        drift = theta * x + grad_w
        noise = torch.randn_like(x) * (2 * D * step_size) ** 0.5
        x = x + step_size * drift + noise

        if i % 10 == 0:
            samples = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0)
            grid = make_grid(samples, nrow=int(np.sqrt(batch_size)))
            frames.append(grid.permute(1, 2, 0).cpu().numpy())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    im = ax.imshow(frames[0], vmin=0.0, vmax=1.0)

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    plt.close(fig)
    return ani


def plot_tsne(samples, samples_init, data_loader, device='cuda'):
    from sklearn.manifold import TSNE
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    real_data, real_labels = next(iter(data_loader))
    real_data = real_data.view(real_data.size(0), -1).detach().cpu()

    fake_data = samples.detach().view(samples.size(0), -1).cpu()
    fake_data_init = samples_init.detach().view(samples_init.size(0), -1).cpu()

    all_data = torch.cat([real_data, fake_data, fake_data_init], dim=0).numpy()
    embeddings = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(all_data)

    n_real = real_data.size(0)
    n_fake = fake_data.size(0)
    real_emb = embeddings[:n_real]
    fake_emb = embeddings[n_real:n_real + n_fake]
    fake_emb_init = embeddings[n_real + n_fake:]

    plt.figure(figsize=(10, 8))
    for digit in range(10):
        idx = (real_labels == digit).numpy()
        plt.scatter(real_emb[idx, 0], real_emb[idx, 1], s=10, alpha=0.5, label=f'Digit {digit}')
    plt.scatter(fake_emb[:, 0], fake_emb[:, 1], s=10, c='black', alpha=0.5, label='Generated')
    plt.scatter(fake_emb_init[:, 0], fake_emb_init[:, 1], s=10, c='green', alpha=0.9, label='Initial')
    plt.title('t-SNE: Real MNIST vs Generated')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
