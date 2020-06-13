# Generate data

import jax.random as random
import jax.numpy as np
import numpyro.distributions as dist
import matplotlib.pyplot as plt

rng_key1, rng_key2 = random.split(random.PRNGKey(42))

N = 50
X = dist.Uniform(0.0, 5.0).sample(rng_key1, sample_shape=(N,1))

noise = dist.Uniform(0, 0.5).sample(rng_key2, sample_shape=(X.shape[0],))
y = 0.5 * np.sin(3 * X[:,0]) + noise

plt.plot(X,y,'*')