import jax.random as random
import numpyro.distributions as dist
import jax.numpy as np


def generate_sin(n_obs, noise_level, seed):
    """
    Generate data (noisy and non-noisy from 0.5*sin(3*x)
    :param n_obs:
    :param noise_level:
    :return: tuple of noisy data and original data (without noise)
            data = {'X': X, 'y': y}
    """
    # jax random generator
    rng_key1, rng_key2 = random.split(random.PRNGKey(seed))
    # uniform sample from X
    X = dist.Uniform(0.0, 5.0).sample(rng_key1, sample_shape=(n_obs, 1))
    # generate noise
    noise = dist.Uniform(-noise_level, noise_level).sample(
        rng_key2, sample_shape=(X.shape[0],)
    )
    # generate y
    y = 0.5 * np.sin(3 * X[:, 0]) + noise
    noisy_data = {
        "X": X,
        "y": y,
    }

    # generate real observation from 0.5*sin(3*x)
    X_ = np.linspace(0.0, 5.0, 400)
    y_ = 0.5 * np.sin(3 * X_)
    data = {
        "X": X_,
        "y": y_,
    }

    return noisy_data, data
