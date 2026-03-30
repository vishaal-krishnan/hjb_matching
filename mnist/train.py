import torch
import torch.nn as nn


def OU_sample(x, t, theta, D):
    device = x.device
    t_batched = torch.ones(x.shape[0], device=device) * t
    exp_term = torch.exp(-theta * t_batched).view(-1, 1, 1, 1)
    mean = exp_term * x
    std = torch.sqrt((D / theta) * (1.0 - exp_term ** 2)).view(-1, 1, 1, 1)
    return mean + std * torch.randn_like(x)


def grad_w_fn(model, x, t):
    x = x.detach().requires_grad_(True)
    output = model(x, t.detach())
    grad_w = torch.autograd.grad(
        outputs=output.sum(), inputs=x,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad_w


def spatial_smoothness(w, kernel_size=3):
    grad_x = w[:, :, :, 1:] - w[:, :, :, :-1]
    grad_y = w[:, :, 1:, :] - w[:, :, :-1, :]
    return (grad_x ** 2).mean() + (grad_y ** 2).mean()


def loss_fn(model, x, t, cfg, eps=1e-5):
    beta = cfg['beta']
    theta = cfg['theta']
    D = cfg['D']
    device = x.device
    batch_size = x.shape[0]

    random_t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
    random_t_neg = torch.rand(batch_size, device=device) * (1.0 - eps) + eps

    exp_term = torch.exp(-theta * random_t).view(-1, 1, 1, 1)
    mean = exp_term * x
    std = torch.sqrt((D / theta) * (1.0 - exp_term ** 2)).view(-1, 1, 1, 1)
    x_t = mean + std * torch.randn_like(x)

    w = model(x_t, random_t)
    z = torch.exp(beta * w)

    w0 = model(x, torch.zeros_like(random_t))
    z0 = torch.exp(beta * w0)

    loss_fk = (z - torch.exp(-beta * random_t) * z0) ** 2
    loss_dual = (z0 - 1.0) ** 2

    w_neg = model(x_t, random_t_neg)
    z_neg = torch.exp(beta * w_neg)
    loss_neg = (z_neg - 0.1) ** 2

    loss_spatial = spatial_smoothness(x_t)

    total = 1e0 * loss_fk + 1e0 * loss_dual + 1e-3 * loss_spatial + 1e-1 * loss_neg
    return total.mean(), loss_fk.mean(), loss_dual.mean()


def Euler_Maruyama_sampler(model, cfg, batch_size, device='cuda'):
    theta = cfg['theta']
    D = cfg['D']
    num_steps = cfg['num_steps']

    time_steps = torch.linspace(1.0, 0.0, steps=num_steps + 2, device=device)[1:-1]
    step_size = time_steps[0] - time_steps[1]

    x = torch.randn(batch_size, 1, 28, 28, device=device) * (D / theta) ** 0.5
    init_x = x.clone()

    model.eval()
    with torch.no_grad():
        for s in time_steps:
            batch_t = torch.ones(batch_size, device=device) * s
            grad_w = grad_w_fn(model, x, batch_t)
            drift = theta * x + grad_w
            noise = torch.randn_like(x) * (2 * D * step_size) ** 0.5
            x = x + step_size * drift + noise

    return x, init_x


def train(model, optimizer, data_loader, cfg, device='cuda'):
    n_epochs = cfg['n_epochs']
    log_every = cfg.get('log_every', 10)

    loss_lst, loss_fk_lst, loss_dual_lst = [], [], []

    for epoch in range(n_epochs):
        avg_loss = avg_fk = avg_dual = 0.0
        num_items = 0

        model.train()
        for x, _ in data_loader:
            x = x.to(device)
            t = torch.rand(1, device=device)

            loss, lf, ld = loss_fn(model, x, t, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = x.shape[0]
            avg_loss += loss.item() * n
            avg_fk += lf.item() * n
            avg_dual += ld.item() * n
            num_items += n

        loss_lst.append(avg_loss / num_items)
        loss_fk_lst.append(avg_fk / num_items)
        loss_dual_lst.append(avg_dual / num_items)

        print(f"Epoch {epoch:3d}  loss={loss_lst[-1]:.6f}  fk={loss_fk_lst[-1]:.6f}  dual={loss_dual_lst[-1]:.6f}")

        if epoch % log_every == 0:
            yield epoch, model, loss_lst, loss_fk_lst, loss_dual_lst

    yield n_epochs - 1, model, loss_lst, loss_fk_lst, loss_dual_lst
