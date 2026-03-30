import argparse
import yaml
import jax
import jax.numpy as jnp
import optax

from toy.distributions import DISTRIBUTIONS, nu_lensing
from toy.model import build_w_net
from toy.train import build_train_fns
from toy.plot import plot_training_progress, animate_w_field, make_video, save_combined_snapshots


def main():
    parser = argparse.ArgumentParser(description="Run a toy HJB matching experiment.")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    N = cfg['N']
    key = jax.random.PRNGKey(0)

    # --- Distribution ---
    init_sample = DISTRIBUTIONS[cfg['distribution']]
    nu_fn = nu_lensing if cfg.get('use_analytical_nu', False) else None
    target_pos = jnp.array([cfg['target_pos']])

    # --- Model ---
    w_net = build_w_net(cfg)
    key, subkey = jax.random.split(key)
    w_params = w_net.init(subkey, jnp.zeros((N, 2)), jnp.full((N,), 0, dtype=jnp.int32))

    # --- Optimizer ---
    w_opt = optax.adam(cfg['lr'])
    w_opt_state = w_opt.init(w_params)

    # --- Training functions ---
    fns = build_train_fns(w_net, w_opt, cfg, nu_fn=nu_fn)
    train_step = fns['train_step']
    rollout = fns['rollout']

    # --- Training loop ---
    loss_lst = []
    log_every = cfg.get('log_every', 50)

    for epoch in range(cfg['total_epochs']):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        pos0 = init_sample(subkey1, N)

        w_params, w_opt_state, loss, loss1, loss2, loss3, final_pos, _ = train_step(
            w_params, w_opt_state, pos0, subkey2, target_pos
        )
        loss_lst.append(float(loss))

        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d}  loss={loss:.6f}  loss1={loss1:.6f}  loss2={loss2:.6f}  loss3={loss3:.6f}")
            plot_training_progress(epoch, loss_lst, final_pos, pos0, target_pos,
                                   w_net, w_params, cfg, nu_fn=nu_fn)

    # --- Visualization ---
    key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 7)

    animate_w_field(w_net, w_params, cfg)

    make_video(rollout, init_sample, w_params, cfg, subkey1, subkey2,
               step_mode="focusing", nu_fn=nu_fn)

    make_video(rollout, init_sample, w_params, cfg, subkey3, subkey4,
               step_mode="reversing", nu_fn=nu_fn)

    save_combined_snapshots(w_net, rollout, w_params, cfg, subkey5, subkey6)


if __name__ == '__main__':
    main()
