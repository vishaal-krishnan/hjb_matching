import jax
import jax.numpy as jnp


def four_gaussian(key, N, means=[(2.0, 0.0), (0.0, 2.0), (0.0, -2.0), (-2.0, 0.0)], std=0.2):
    keys = jax.random.split(key, 4)
    N_per_cluster = N // 4
    clusters = [
        jax.random.normal(keys[i], (N_per_cluster, 2)) * std + jnp.array(means[i])
        for i in range(4)
    ]
    return jnp.concatenate(clusters, axis=0)


def two_moon(key, N, noise=0.1):
    N_half = N // 2
    theta1 = jnp.linspace(0, jnp.pi, N_half)
    theta2 = jnp.linspace(0, jnp.pi, N_half)
    x1 = jnp.stack([2.0 * jnp.cos(theta1), 2.0 * jnp.sin(theta1) + 1.0], axis=1)
    x2 = jnp.stack([2.0 - 2.0 * jnp.cos(theta2), -2.0 * jnp.sin(theta2) - 1.0], axis=1)
    data = jnp.concatenate([x1, x2], axis=0)
    noise_key = jax.random.split(key, 1)[0]
    data += noise * jax.random.normal(noise_key, data.shape)
    return data


def swissroll(key, N, noise=0.05):
    t = jax.random.uniform(key, shape=(N,), minval=jnp.pi, maxval=5.5 * jnp.pi)
    x = 0.2 * t * jnp.cos(t)
    y = 0.2 * t * jnp.sin(t)
    data = jnp.stack([x, y], axis=1)
    noise_key = jax.random.split(key, 1)[0]
    data += noise * jax.random.normal(noise_key, data.shape)
    return data


def single_gaussian(key, N, mean=(-1.0, 0.0), std=0.1):
    return jax.random.normal(key, (N, 2)) * std + jnp.array(mean)


def nu_lensing(x, means=[(0.0, 0.0)], std=0.1):
    x = jnp.atleast_2d(x)
    nu = jnp.zeros(x.shape[0])
    for mean in means:
        diff = (x - jnp.array(mean)) / std
        exponent = -0.5 * jnp.sum(diff ** 2, axis=-1)
        nu = nu + jnp.exp(exponent)
    return 1.0 - 20 * nu


DISTRIBUTIONS = {
    'four_gaussian': four_gaussian,
    'two_moon': two_moon,
    'swissroll': swissroll,
    'single_gaussian': single_gaussian,
}
