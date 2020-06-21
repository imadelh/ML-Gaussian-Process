"""
Variational Inference
"""
import jax.random as random
import jax.numpy as np
from jax import lax
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from gpr import GPRegression, RBFKernel, mcmc_inference


import numpyro.optim as optim
from numpyro.infer import SVI
from numpyro.infer import ELBO


# data


rng_key1, rng_key2 = random.split(random.PRNGKey(42))

N = 50
X = dist.Uniform(0.0, 5.0).sample(rng_key1, sample_shape=(N,1))

noise = dist.Uniform(0.0, 0.5).sample(rng_key2, sample_shape=(X.shape[0],))
y = 0.5 * np.sin(3 * X[:,0]) + noise


#MLE

prior = {
    'length': 1.0,
    'var': 1.0,
    'noise': 1.0,
}


def guide(X, Y):
    return None

gp = GPRegression(RBFKernel, prior, guide)

svi = SVI(gp.likelihood, gp.guide, optim.Adam(0.005), ELBO(num_particles=1), X=X, Y=y)

svi_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), svi_state, np.zeros(2500))


plt.plot(loss)
plt.savefig('plots/loss.png')


# MLE estimates
params = svi.get_params(state)
for name, value in params.items():
  print("{} = {}".format(name, value))