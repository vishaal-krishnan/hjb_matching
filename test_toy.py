"""Quick smoke test: runs 5 epochs of each toy config and reports results."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed

import yaml
import jax
import jax.numpy as jnp
import optax

from toy.distributions import DISTRIBUTIONS, nu_lensing
from toy.model import build_w_net
from toy.train import build_train_fns

CONFIGS = [
    ('4gaussian',  'toy/configs/4gaussian.yaml'),
    ('two_moon',   'toy/configs/two_moon.yaml'),
    ('swissroll',  'toy/configs/swissroll.yaml'),
    ('lensing',    'toy/configs/lensing.yaml'),
]
TEST_EPOCHS = 5


def run_config(name, config_path):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    N = cfg['N']
    key = jax.random.PRNGKey(0)

    init_sample = DISTRIBUTIONS[cfg['distribution']]
    nu_fn = nu_lensing if cfg.get('use_analytical_nu', False) else None
    target_pos = jnp.array([cfg['target_pos']])

    w_net = build_w_net(cfg)
    key, subkey = jax.random.split(key)
    w_params = w_net.init(subkey, jnp.zeros((N, 2)), jnp.full((N,), 0, dtype=jnp.int32))

    w_opt = optax.adam(cfg['lr'])
    w_opt_state = w_opt.init(w_params)

    fns = build_train_fns(w_net, w_opt, cfg, nu_fn=nu_fn)
    train_step = fns['train_step']
    rollout    = fns['rollout']

    for epoch in range(TEST_EPOCHS):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        pos0 = init_sample(subkey1, N)

        w_params, w_opt_state, loss, loss1, loss2, loss3, final_pos, traj = train_step(
            w_params, w_opt_state, pos0, subkey2, target_pos
        )
        print(f"  epoch {epoch}  loss={float(loss):.6f}  loss1={float(loss1):.6f}  loss2={float(loss2):.6f}  loss3={float(loss3):.6f}")

    # Also test rollout (used by make_video / save_combined_snapshots)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    pos0 = init_sample(subkey1, N)
    step_idx_vec, dt_vec, traj, traj_w, traj_nu = rollout(
        pos0, subkey2, w_params, None, "focusing", "pretraining", target_pos, 0
    )
    print(f"  rollout (focusing) OK — traj shape: {traj.shape}")

    key, subkey1, subkey2 = jax.random.split(key, 3)
    pos0_rev = jax.random.normal(subkey1, (N, 2)) * cfg.get('reverse_init_std', 0.25) + target_pos
    step_idx_vec, dt_vec, traj, traj_w, traj_nu = rollout(
        pos0_rev, subkey2, w_params, None, "reversing", "pretraining", target_pos, 0
    )
    print(f"  rollout (reversing) OK — traj shape: {traj.shape}")
    print(f"  PASSED")


if __name__ == '__main__':
    for name, path in CONFIGS:
        run_config(name, path)

    print(f"\n{'='*50}")
    print("  All configs passed.")
    print(f"{'='*50}\n")
