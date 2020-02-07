# Neural networks on circular data distributions 

<img src="https://raw.githubusercontent.com/njkrichardson/fouriernets/master/figs/circular_distributions.png" alt="drawing" height="200" width="300" class="center"/>


This package uses tooling from autograd to implement various neural nets toward discriminatively modeling distributions over circular data. I intend to show examples using multilayer perceptrons, autoencoders, and variational autoencoders (currently a tutorial for generating a simple mlp is available). 

Coming Soon: 
  * autoencoders
  * variational autoencoders 
  * Bayesian neural nets 
  * Bayesian optimization utility 
  * 3D data over the sphere

_Author: [Nick Richardson](https://github.com/njkrichardson)_


References: https://scholarship.claremont.edu/hmc_theses/226/

## Installing from git source

```
git clone git@github.com:njkrichardson/fouriernets.git
pip install -e fouriernets
```

# Background 

Modeling natural data (images, speech, language) is often a task of identifying label-dependent invariances that allow us to discriminate between the different classes even on unseen inputs. These invariants reflect the latent structure in the data. Data over circular and spherical manifolds arise as a special case of rotational structure. Discriminating between different distributions over circles and spheres then, is an interesting task of density estimation in which one hopes to capture and represent the inherent rotational symmetry in the data. 

The package provides a number of utilities to generate distributions over circles, and create a variety of neural networks to discriminatively model the distributions. 

# Example 

The fouriernets package provides a number of utilities to create and visualize distributions over circles, or make your own dataset: the distribution metaparameters can be provided as kwargs for custom datasets. 

```python 
from utils import make_data, plot_sample, train_test_split

# simulation parameters 
n_draws = 128   # number of draws from each distribution 
n_bins = 64     # number of bins in each distribution 
n_data = 1000   # number of data per class

# generate the data 
train_inputs, test_inputs, train_labels, test_labels = make_data(n_per_class=n_data, n_bins=n_bins, n_draws=n_draws, split=True)
```

fouriernets also provides simple plotting and visualization utilities 

```python
plot_sample(train_inputs, train_labels)
```

![sample_fig](figs/samples.png)

you can then instantiate arbitrary neural network models using autograd. These functions and others are provided 
in nnet.py 

```python 
# basic mlp 
def init_mlp_params(scale, layer_sizes, rs=npr.RandomState(0)):
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def mlp_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def l2_norm(params):
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def mlp_log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(mlp_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(mlp_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)
```

it's then just as easy as defining an objective function and its gradient with respect to the parameters, and then training the network. 

```python 
def objective(params, iter):
    idx = batch_indices(iter)
    return -mlp_log_posterior(params, train_inputs[idx], train_labels[idx], L2_reg)

# get gradient of objective using autograd
objective_grad = grad(objective)

# The optimizers provided can optimize lists, tuples, or dicts of parameters.
optimized_params = adam(objective_grad, init_params, step_size=step_size,
                        num_iters=num_epochs * num_batches)
```
![sample_fig](figs/training_performance.png)

additional diagnostic tools, models, and tutorials are forthcoming. 
