import jax.random as random
import jax.numpy as np
from jax import vmap
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from gpr import GPRegression, RBFKernel, mcmc_inference

rng_key1, rng_key2 = random.split(random.PRNGKey(42))

N = 50
X = dist.Uniform(0.0, 5.0).sample(rng_key1, sample_shape=(N,1))

noise = dist.Uniform(-0.2, 0.2).sample(rng_key2, sample_shape=(X.shape[0],))
y = 0.5 * np.sin(3 * X[:,0]) + noise


plt.plot(X, y,'kx', color='r', label='data')
X_ = np.linspace(0.0, 5.0, 400)
plt.plot(X_, 0.5 * np.sin(3 * X_), color='navy', label='True')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Noisy data')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})

plt.savefig('plots/image1.png')
plt.close()


prior = {
    'length': dist.LogNormal(0.0, 1.0),
    'var': dist.LogNormal(0.0, 1.0),
    'noise': dist.LogNormal(0.0, 1.0),
}

gp = GPRegression(RBFKernel, prior, None)

# Plot GP prior before inference
#TODO
#Apply Kernel and get sa,ples from normal distribution
# Like pyro https://pyro.ai/examples/gp.html
# Or use predictive dist
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
# https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/gaussian_process/_gpr.py#L381
# Sample params from prior
# Do prediction
# REturn (mean, covariance matric)
# sample from the normal distribution associated with it


rng_key, rng_key_predict = random.split(random.PRNGKey(0))

num_warmup=200
num_samples=200
num_chains=1

# do inference
samples = mcmc_inference(gp.likelihood, num_warmup, num_samples, num_chains, rng_key, X, y)


# Plot GP post after inference
#TODO


# Get samples of params
vmap_args = []
params_keys = ['var', 'length', 'noise']

for i in range(num_samples*num_chains):
  params_dict = []
  for name in params_keys:
    params_dict.append(samples[name][i])

  vmap_args.append(np.array(params_dict))

vmap_args = (np.array(vmap_args),)

# Run predictions
X_test = np.linspace(-1, 6, 400)[:,None] # 400 points where to compute the GP
means, sigma = vmap(lambda params:
                          gp.predict(X, y, X_test, params_keys, params))(*vmap_args)

mean_prediction = np.mean(means, axis=0)
sigma_predictions = np.mean(sigma, axis=0 )

# make plots
fig, ax = plt.subplots(1, 1)

# plot training data
ax.plot(X, y, 'kx')
# plot 90% confidence level of predictions
X_test = np.linspace(-1, 6, 400)
ax.fill_between(X_test, mean_prediction - 2.0 * sigma_predictions, mean_prediction + 2.0 * sigma_predictions, color='lightblue',)
# plot mean prediction
ax.plot(X_test, mean_prediction, 'blue', ls='solid', lw=2.0)
ax.plot(X_, 0.5 * np.sin(3 * X_), color='navy', label='True')

ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")
plt.savefig('plots/pred.png')