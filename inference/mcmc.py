from numpyro.infer import MCMC, NUTS, HMC
import time


def mcmc_inference(model, num_warmup, num_samples, num_chains, rng_key, X, Y):
    """"
    Helper function for doing NUTS inference.
    :param model: a parametric function proportional to the posterior (see gp_regression.likelihood).
    :param num_warmup: warmup steps.
    :param num_samples: number of samples.
    :param num_chains: number of Markov chains used for MCMC sampling.
    :param rng_key: random seed.
    :param X: X data.
    :param Y: Y data.
    :return: Dictionary key: name of parameter (from defined in model), value: list of samples.
    """
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=num_chains)
    mcmc.run(rng_key, X, Y)
    print('\nMCMC time:', time.time() - start)
    print(mcmc.print_summary())
    return mcmc.get_samples()