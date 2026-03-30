import jax
import jax.numpy as jnp
import haiku as hk


def w_net_positional_fn(hidden_dim=64, num_layers=8, num_steps=100):
    class PosEmbed(hk.Module):
        def __init__(self, num_steps, dim, name=None):
            super().__init__(name=name)
            self.num_steps = num_steps
            self.dim = dim

        def __call__(self, positions):
            embed_table = hk.get_parameter(
                "pos_embed", [self.num_steps, self.dim],
                init=hk.initializers.TruncatedNormal(stddev=0.02)
            )
            return embed_table[positions]

    def net_fn(x, step_expanded):
        x_embed = hk.nets.MLP([hidden_dim, hidden_dim])(x)
        step_embed = PosEmbed(num_steps=num_steps, dim=hidden_dim)(step_expanded)
        h = x_embed + step_embed
        for _ in range(num_layers):
            shortcut = h
            h = hk.Linear(hidden_dim)(h)
            h = jax.nn.gelu(h)
            h = hk.Linear(hidden_dim)(h)
            if shortcut.shape[-1] != hidden_dim:
                shortcut = hk.Linear(hidden_dim)(shortcut)
            h = h + shortcut
        return hk.Linear(1)(h)

    return net_fn


def w_net_sinusoidal_fn(hidden_dim=64, num_layers=10, num_steps=100):
    def sinusoidal_embedding(step_ids, dim):
        step_ids = step_ids.astype(jnp.float32)[:, None]
        freqs = jnp.exp(-jnp.arange(0, dim, 2) * (jnp.log(10000.0) / dim))
        angles = step_ids * freqs
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    def net_fn(x, step_expanded):
        x_embed = hk.nets.MLP([hidden_dim, hidden_dim])(x)
        step_embed = sinusoidal_embedding(step_expanded, hidden_dim)
        h = x_embed + step_embed
        for _ in range(num_layers):
            h = hk.Linear(hidden_dim)(h)
            h = jax.nn.gelu(h)
            h += step_embed
        return hk.Linear(1)(h)

    return net_fn


def nu_net_fn(hidden_dim=128, num_layers=10):
    def nu_net(x):
        h = x
        for _ in range(num_layers):
            h = hk.Linear(hidden_dim)(h)
            h = jax.nn.gelu(h)
        h = hk.Linear(1)(h)
        return 1.0 + 0.5 * jax.nn.tanh(h)
    return nu_net


def build_w_net(cfg):
    variant = cfg.get('w_net', 'positional')
    num_steps = cfg['steps']
    if variant == 'positional':
        fn = w_net_positional_fn(num_steps=num_steps)
    elif variant == 'sinusoidal':
        fn = w_net_sinusoidal_fn(num_steps=num_steps)
    else:
        raise ValueError(f"Unknown w_net variant: {variant}")
    return hk.without_apply_rng(hk.transform(fn))


def compute_grad(w_net, w_params, x, idx):
    def single_forward(x_single, idx_single):
        x_single = x_single[None, :]
        idx_single = idx_single[None]
        return w_net.apply(w_params, x_single, idx_single).squeeze()
    grad_fn = jax.vmap(jax.grad(single_forward), in_axes=(0, None))
    return grad_fn(x, idx)
