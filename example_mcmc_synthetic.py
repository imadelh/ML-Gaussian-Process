"""
Gaussian Process Regression with MCMC posterior simulation.
Synthetic data
"""
import jax.random as random
import jax.numpy as np
from jax import vmap
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as npp
from gpr import GPRegression, RBFKernel, generate_sin, get_logger
from inference import mcmc_inference
from collections import namedtuple
import seaborn as sns

plt.style.use("seaborn-colorblind")
plt.style.use("seaborn-whitegrid")

configuration = {
    "exp_name": "Simulated data",
    "data": "simulated",
    "seed": 42,
    # in case of simulated data
    "n_obs": 50,
    "noise_level": 0.2,
    # GP parameters
    "kernel": RBFKernel,
    "priors": {
        # same names as in RBFKernel
        "length": dist.LogNormal(0.0, 1.0),
        "var": dist.LogNormal(0.0, 1.0),
        "noise": dist.LogNormal(0.0, 1.0),
    },
    # MCMC parameters
    "num_warmup": 200,
    "num_samples": 200,
    "num_chains": 1,
}


def main():
    # Read configuration
    MyTuple = namedtuple("configuration", sorted(configuration))
    config = MyTuple(**configuration)

    # Define seed and log file
    logger = get_logger("results/mcmc/mcmc_{}.log".format(config.exp_name))

    seed = config.seed
    npp.random.seed(seed)

    # 1. Generate toy data
    logger.info("Generate simulated data")
    noisy_data, real_data = generate_sin(
        n_obs=config.n_obs, noise_level=config.noise_level, seed=seed
    )
    plt.figure(figsize=(16, 9))
    plt.plot(noisy_data["X"], noisy_data["y"], "kx", label="Observations")
    plt.plot(real_data["X"], real_data["y"], label="Ground truth")
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Noisy data")
    plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.savefig("results/mcmc/data.png")
    plt.close()
    logger.info("Plot generated data")

    # 2. Plot prior functions
    logger.info("Plot prior functions with RBF(1,1)")
    X_ = np.linspace(-1, 6, 500)[:, None]
    kernel_params = {
        "length": 1.0,
        "var": 1.0,
    }
    y_cov = config.kernel(X_, X_, kernel_params)
    y_mean = np.zeros(X_.shape[0])

    plt.figure(figsize=(16, 9))
    for i in range(10):
        y_samples = npp.random.multivariate_normal(y_mean, y_cov, 1, tol=1e-5).T
        plt.plot(X_, y_samples)

    # Plot mean/variance
    y_std = np.sqrt(np.diag(y_cov))
    plt.plot(X_, y_mean, "k", lw=3, zorder=9)
    plt.fill_between(
        X_[:, 0], y_mean - y_std, y_mean + y_std, alpha=0.2, color="lightblue"
    )
    plt.title("Priors (kernel: RBF(var=1.0, length=1.0))")
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.savefig("results/mcmc/kernel_priors.png")
    plt.close()

    # 3. Define GPRegression using the kernel and priors over parameters
    gp = GPRegression(config.kernel, config.priors, None)

    # 4. MCMC inference
    logger.info("Running MCMC")
    rng_key = random.PRNGKey(seed)

    # run inference
    samples = mcmc_inference(
        gp.likelihood,
        config.num_warmup,
        config.num_samples,
        config.num_chains,
        rng_key,
        noisy_data["X"],
        noisy_data["y"],
    )
    logger.info("MCMC complete")

    # Plot prior/posterior distribution over parameters
    for name, support in samples.items():
        plt.figure(figsize=(16, 9))
        # Sample from prior distribution
        prior = config.priors[name].sample(rng_key, (500,))
        sns.distplot(prior, label="prior distribution")
        # 'support' is the result of the MCMC simulations
        sns.distplot(support, label="posterior distribution")
        plt.title(name)
        plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
        plt.grid(True, color="#93a1a1", alpha=0.3)
        plt.savefig(f"results/mcmc/dist_{name}.png")
        plt.close()

    vmap_args = []
    params_keys = config.priors.keys()

    for i in range(config.num_samples * config.num_chains):
        params_dict = []
        for name in params_keys:
            params_dict.append(samples[name][i])

        vmap_args.append(np.array(params_dict))

    vmap_args = (np.array(vmap_args),)

    # Run predictions
    logger.info("Posterior predictions on the interval of simulated data")
    X_test = np.linspace(-1, 6, 500)[:, None]  # X points where to compute the GP
    means, sigma = vmap(
        lambda params: gp.predict(
            noisy_data["X"], noisy_data["y"], X_test, params_keys, params
        )
    )(*vmap_args)

    mean_prediction = np.mean(means, axis=0)
    sigma_predictions = np.mean(sigma, axis=0)

    # plot results
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 9)

    # plot training data
    ax.plot(noisy_data["X"], noisy_data["y"], "kx", label="Observations")
    # plot 90% confidence level of predictions
    ax.fill_between(
        X_test[:, 0],
        mean_prediction - 2.0 * sigma_predictions,
        mean_prediction + 2.0 * sigma_predictions,
        color="lightblue",
        label="90% Confidence interval",
    )
    # plot mean prediction
    ax.plot(
        X_test[:, 0],
        mean_prediction,
        "blue",
        ls="solid",
        lw=2.0,
        label="GP posterior mean predictions",
    )

    ax.plot(real_data["X"], real_data["y"], label="True")

    for i in range(10):
        kernel_params = dict(zip(config.priors.keys(), vmap_args[0][i]))
        y_cov = config.kernel(X_test, X_test, kernel_params)
        y_samples = npp.random.multivariate_normal(means[i], y_cov, 1, tol=1e-5).T
        # tol to solve stability problems
        # https://stackoverflow.com/questions/49624840/confusing-behavior-of-np-random-multivariate-normal
        plt.plot(X_test[:, 0], y_samples, alpha=0.5)
    ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")
    plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.savefig("results/mcmc/predictions.png")
    plt.close()


if __name__ == "__main__":
    main()
