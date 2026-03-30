import jax
import jax.numpy as jnp
import optax
from functools import partial

from toy.model import compute_grad


def build_train_fns(w_net, w_opt, cfg, nu_net=None, nu_fn=None):
    """Build JIT-compiled training functions as closures over w_net, w_opt, and cfg.

    Args:
        w_net:   Haiku transformed value network.
        w_opt:   Optax optimizer for w_net.
        cfg:     Config dict.
        nu_net:  Optional Haiku transformed nu network (for finetuning).
        nu_fn:   Optional analytical nu function, e.g. nu_lensing (for lensing experiment).

    Returns:
        Dict with keys: train_step, rollout, loss_fk, loss_fk_local.
    """
    N = cfg['N']
    D = cfg['D']
    beta = cfg['beta']
    guidance_strength = cfg['guidance_strength']
    steps = cfg['steps']
    T_tot = cfg['T_tot']
    ns_minval = cfg.get('ns_minval', -5.0)
    ns_maxval = cfg.get('ns_maxval', 5.0)

    @jax.jit
    def loss_fk(w_params, nu_params, trajectory, step_idx_vec, dt_vec):
        n_steps, batch_size, _ = trajectory.shape

        x_k = trajectory[:-1]
        x_kp1 = trajectory[1:]
        dx = x_kp1 - x_k
        s_k = step_idx_vec[:-1][:, None].repeat(batch_size, axis=1)
        dt_k = dt_vec[:-1][:, None].repeat(batch_size, axis=1)

        xk_flat = x_k.reshape(-1, 2)
        s_k_flat = s_k.reshape(-1)
        dt_k_flat = dt_k.reshape(-1)

        W_k = w_net.apply(w_params, xk_flat, s_k_flat)
        Z_k = jnp.exp(beta * W_k).squeeze()

        x_0_repeated = jnp.repeat(trajectory[0], n_steps - 1, axis=0)
        W_0_repeated = w_net.apply(w_params, x_0_repeated, jnp.zeros_like(s_k_flat))
        Z_0_repeated = jnp.exp(beta * W_0_repeated).squeeze()

        if nu_params is None and nu_fn is not None:
            nu_k = nu_fn(xk_flat)
        elif nu_params is None:
            nu_k = jnp.ones((xk_flat.shape[0],))
        else:
            nu_k = nu_net.apply(nu_params, xk_flat).squeeze()

        cost_single = (nu_k * dt_k_flat).reshape(n_steps - 1, batch_size)
        cost_flat = (-beta * jnp.cumsum(cost_single, axis=0)).reshape(-1)

        x_mid = 0.5 * (x_k + x_kp1)
        p1_flat = ((1.0 / 2.0 * D) * jnp.cumsum(
            guidance_strength * (x_mid * dx).sum(axis=-1), axis=0
        )).reshape(-1)
        p2_flat = (-(1.0 / 4.0 * D) * jnp.cumsum(
            guidance_strength ** 2 * (x_mid ** 2 * dt_k[..., None]).sum(axis=-1), axis=0
        )).reshape(-1)

        target = jnp.exp(cost_flat) * Z_0_repeated
        return jnp.mean((Z_k - target) ** 2)

    @jax.jit
    def loss_fk_local(w_params, nu_params, trajectory, step_idx_vec, dt_vec):
        n_steps, batch_size, _ = trajectory.shape

        x_k = trajectory[:-1]
        x_kp1 = trajectory[1:]
        s_k = step_idx_vec[:-1][:, None].repeat(batch_size, axis=1)
        s_kp1 = step_idx_vec[1:][:, None].repeat(batch_size, axis=1)
        dt_k = dt_vec[:-1][:, None].repeat(batch_size, axis=1)
        dt_kp1 = dt_vec[1:][:, None].repeat(batch_size, axis=1)

        xk_flat = x_k.reshape(-1, 2)
        xkp1_flat = x_kp1.reshape(-1, 2)
        s_k_flat = s_k.reshape(-1)
        s_kp1_flat = s_kp1.reshape(-1)
        dt_k_flat = dt_k.reshape(-1)

        Z_k = jnp.exp(beta * w_net.apply(w_params, xk_flat, s_k_flat)).squeeze()
        Z_kp1 = jnp.exp(beta * w_net.apply(w_params, xkp1_flat, s_kp1_flat)).squeeze()

        if nu_params is None:
            nu_k = jnp.ones((xk_flat.shape[0],))
        else:
            nu_k = nu_net.apply(nu_params, xk_flat).squeeze()

        cost = beta * nu_k * dt_k_flat

        dx = x_kp1 - x_k
        x_mid = 0.5 * (x_k + x_kp1)
        dt_mid = 0.5 * (dt_k + dt_kp1)
        p1 = (-(1.0 / 2.0 * D) * guidance_strength * (x_mid * dx).sum(axis=-1)).reshape(-1)
        p2 = ((1.0 / 4.0 * D) * guidance_strength ** 2 * (x_mid ** 2 * dt_mid[..., None]).sum(axis=-1)).reshape(-1)

        target = jnp.exp(cost + p1 + p2) * Z_kp1
        return jnp.mean((Z_k - target) ** 2)

    def step(pos, target_pos, key, w_params, nu_params, step_idx, dt, step_mode, train_mode):
        step_idx_expanded = jnp.full((pos.shape[0],), step_idx)
        w = w_net.apply(w_params, pos, step_idx_expanded)

        if train_mode == "pretraining":
            nu = jnp.ones_like(w)
        elif train_mode == "finetuning":
            nu = nu_net.apply(nu_params, pos)
        else:
            raise ValueError(f"Invalid train_mode: {train_mode}")

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (N, 2)) * jnp.sqrt(2 * D * dt)
        guidance_vec = target_pos - pos

        if step_mode == "focusing":
            dpos = dt * guidance_strength * guidance_vec + noise
        elif step_mode == "reversing":
            grad_w = compute_grad(w_net, w_params, pos, step_idx)
            dpos = grad_w * dt - dt * guidance_strength * guidance_vec + noise
        else:
            raise ValueError(f"Invalid step_mode: {step_mode}")

        return pos + dpos, key, w, nu

    @partial(jax.jit, static_argnums=(4, 5, 7))
    def rollout(pos0, key, w_params, nu_params, step_mode, train_mode, target_pos, steps_beyond_horizon=0):
        def body(state, inputs):
            step_idx, dt = inputs
            pos, key = state
            new_pos, new_key, w, nu = step(
                pos, target_pos, key, w_params, nu_params, step_idx, dt, step_mode, train_mode
            )
            return (new_pos, new_key), (new_pos, w, nu)

        step_idx_vec = jnp.arange(steps)
        dt_vec = jnp.ones_like(step_idx_vec) / steps * T_tot

        if step_mode == "focusing":
            pass
        elif step_mode == "reversing":
            step_idx_vec = jnp.flip(step_idx_vec[:-1])
            dt_vec = jnp.flip(dt_vec[:-1])
            if steps_beyond_horizon > 0:
                step_idx_vec = jnp.concatenate(
                    [step_idx_vec, jnp.repeat(step_idx_vec[-1], steps_beyond_horizon)]
                )
                dt_vec = jnp.concatenate(
                    [dt_vec, jnp.repeat(dt_vec[-1], steps_beyond_horizon)]
                )
        else:
            raise ValueError(f"Invalid step_mode: {step_mode}")

        (_, _), (traj, traj_w, traj_nu) = jax.lax.scan(body, (pos0, key), (step_idx_vec, dt_vec))
        return step_idx_vec, dt_vec, traj, traj_w, traj_nu

    def forward_loss_fn(w_params, pos0, key, target_pos):
        subkey1, subkey2 = jax.random.split(key)
        step_idx_vec, dt_vec, traj, traj_w, _ = rollout(
            pos0, subkey1, w_params, None, "focusing", "pretraining", target_pos
        )
        final_pos = traj[-1]
        w_0 = traj_w[0]
        w_T = traj_w[-1]

        loss1 = 1e-1 * (jnp.mean(w_T) - jnp.mean(w_0)) + jnp.mean((w_0 - 10.0) ** 2)
        loss2 = loss_fk(w_params, None, traj, step_idx_vec, dt_vec)
        loss3 = loss_fk_local(w_params, None, traj, step_idx_vec, dt_vec)

        pos_ns = jax.random.uniform(
            subkey2, shape=traj.shape, minval=ns_minval, maxval=ns_maxval
        ).reshape(-1, 2)
        step_ns = jnp.arange(steps)[:, None].repeat(traj.shape[1], axis=1).reshape(-1)
        w_ns = w_net.apply(w_params, pos_ns, step_ns)
        penalty1 = jnp.mean(w_ns ** 2)

        loss = 1e-2 * loss1 + loss2 + 0.1 * penalty1
        return loss.squeeze(), (loss1.squeeze(), loss2.squeeze(), loss3.squeeze(), final_pos, traj)

    @jax.jit
    def train_step(w_params, w_opt_state, pos0, key, target_pos):
        (loss, (loss1, loss2, loss3, final_pos, traj)), grads = jax.value_and_grad(
            forward_loss_fn, has_aux=True
        )(w_params, pos0, key, target_pos)
        updates, w_opt_state_new = w_opt.update(grads, w_opt_state)
        w_params_new = optax.apply_updates(w_params, updates)
        return w_params_new, w_opt_state_new, loss, loss1, loss2, loss3, final_pos, traj

    return dict(
        loss_fk=loss_fk,
        loss_fk_local=loss_fk_local,
        rollout=rollout,
        train_step=train_step,
    )
