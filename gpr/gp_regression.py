import jax.numpy as np
import numpyro
import numpyro.distributions as dist


class GPRegression:
    def __init__(self, kernel, params_priors, guide=None, jitter=1.0e-6):
        """
        :param kernel: A positive definite kernel function (see kernels.py for an example)
        :param params_priors: Prior distributions over the parameters of the kernel or noise
        :param guide: Approximation distribution used for SVI
        :param jitter: Positive constant for numerical stability.
        """
        self.kernel = kernel
        self.params_priors = params_priors
        self.guide = guide
        self.jitter = jitter  # For numerical stability

    def likelihood(self, X, Y):
        """
        The likelihood model for Gaussian Process
        :param X: Features
        :param Y: Targets
        :return: Samples from the a Gaussian Process model.
        """
        params_samples = {}
        for param in self.params_priors:
            if "distribution" in str(type(self.params_priors[param])):
                params_samples[param] = numpyro.sample(param, self.params_priors[param])
            if type(self.params_priors[param]) == float:
                params_samples[param] = numpyro.param(param, self.params_priors[param])

        noise = params_samples.get("noise", 0)
        k = self.kernel(X, X, params_samples) + (noise + self.jitter) * np.eye(
            X.shape[0]
        )

        # sample Y according to the standard gaussian process formula
        numpyro.sample(
            "Y",
            dist.MultivariateNormal(loc=np.zeros(X.shape[0]), covariance_matrix=k),
            obs=Y,
        )

    def predict(self, X, Y, X_new, params_keys, params_values):
        """
        Posterior prediction at a new data point X_new.
        NB: Giving params as separated dict, in order to use VMAP on arrays of params_values
        Given one sample from params (found by SVI or MCMC)
        return the the posterior predictive distribution on X_new
        (using cholesky instead of simple inverse)

        :param X: Training data - Features
        :param Y: Training data - Targets
        :param X_new: New data points for predictions
        :param params_keys: Name of parameters (for Kernel and Noise)
        :param params_values: Value of parameters
        :return: Mean and variance of the posterior predictive distribution at X_new.
        """
        # construct params dict
        params = dict(zip(params_keys, params_values))
        noise = params["noise"]
        k_pp = self.kernel(X_new, X_new, params) + (noise + self.jitter) * np.eye(
            X_new.shape[0]
        )

        k_pX = self.kernel(X_new, X, params)
        k_XX = self.kernel(X, X, params) + (noise + self.jitter) * np.eye(X.shape[0])

        c = np.linalg.inv(np.linalg.cholesky(k_XX))
        inverse2 = np.dot(c.T, c)

        K = k_pp - np.matmul(k_pX, np.matmul(inverse2, np.transpose(k_pX)))
        sigma_noise = np.sqrt(np.clip(np.diag(K), a_min=0.0))
        mean = np.matmul(k_pX, np.matmul(inverse2, Y))

        return mean, sigma_noise
