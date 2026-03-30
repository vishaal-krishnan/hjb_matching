import argparse
import yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from mnist.model import ScoreNet
from mnist.train import train, Euler_Maruyama_sampler
from mnist.plot import (
    plot_loss_curves, plot_generated_samples,
    plot_w_trajectories, plot_bump,
    create_generation_video, plot_tsne
)


def main():
    parser = argparse.ArgumentParser(description="Run the MNIST HJB matching experiment.")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get('device', 'cuda')

    # --- Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])
    dataset = MNIST('.', train=True, transform=transform, download=True)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 4)
    )

    # --- Model ---
    model = torch.nn.DataParallel(ScoreNet())
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg['lr'])

    # --- Initial diagnostics ---
    x, _ = next(iter(data_loader))
    x = x.to(device)
    plot_w_trajectories(model, x, cfg['theta'], cfg['D'], device=device)
    plot_bump(model, x, cfg['theta'], cfg['D'], device=device)

    # --- Training ---
    samples = samples_init = None
    for epoch, model, loss_lst, loss_fk_lst, loss_dual_lst in train(model, optimizer, data_loader, cfg, device):
        plot_loss_curves(loss_lst, loss_fk_lst, loss_dual_lst)
        plot_w_trajectories(model, x, cfg['theta'], cfg['D'], device=device)
        plot_bump(model, x, cfg['theta'], cfg['D'], device=device)

        samples, samples_init = Euler_Maruyama_sampler(
            model, cfg, batch_size=cfg.get('sample_batch_size', 16), device=device
        )
        plot_generated_samples(samples)

        torch.save(model.state_dict(), 'ckpt.pth')

    # --- Final visualization ---
    if samples is not None:
        plot_tsne(samples, samples_init, data_loader, device=device)

    ani = create_generation_video(model, cfg, device=device)
    from IPython.display import HTML
    return HTML(ani.to_jshtml())


if __name__ == '__main__':
    main()
