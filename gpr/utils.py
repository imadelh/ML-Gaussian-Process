from numpyro.infer import MCMC, NUTS, HMC
import time

# helper function for doing hmc inference
def mcmc_inference(model, num_warmup, num_samples, num_chains, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=num_chains)
    mcmc.run(rng_key, X, Y)
    print('\nMCMC elapsed time:', time.time() - start)
    print(mcmc.print_summary())
    return mcmc.get_samples()