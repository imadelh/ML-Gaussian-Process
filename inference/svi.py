from jax import lax
import jax.numpy as np
import numpyro.optim as optim
from numpyro.infer import SVI
from numpyro.infer import ELBO


def svi(model, guide, num_steps, lr, rng_key, X, Y):
    """
    Helper function for doing SVI inference.
    """
    svi = SVI(model, guide, optim.Adam(lr), ELBO(num_particles=1), X=X, Y=Y)

    svi_state = svi.init(rng_key)
    print('Optimizing...')
    state, loss = lax.scan(lambda x, i: svi.update(x), svi_state, np.zeros(num_steps))

    return loss, svi.get_params(state)
