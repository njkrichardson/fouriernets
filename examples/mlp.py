from math import pi
from typing import * 

from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns 

from distributions import uniform, linear, semi_circular, von_mises
from utils import make_data, plot_sample, train_test_split, one_hot_decoding, plot_distribution
from nnet import init_mlp_params, mlp_log_posterior
from type_aliases import *

# --- seaborn stylization 
sns.set_style('white')
sns.set(rc={'figure.figsize':(18, 11)})

# --- seed 
npr.seed(0)

# --- basic mlp 
def init_mlp_params(scale: float, layer_sizes: List[int], key=npr.RandomState(0)):
    return [(scale * key.randn(m, n),   
             scale * key.randn(n))     
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def mlp_predict(params: list, inputs: ndarray) -> ndarray:
    for weight, bias in params:
        outputs: ndarray = np.dot(inputs, weight) + bias
        inputs: ndarray = np.tanh(outputs)

    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def norm(params: list) -> float:
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def mlp_log_posterior(params: list, inputs: ndarray, targets: ndarray, regularization: float):
    log_prior: float = -regularization * norm(params)
    log_likelihood: float = np.sum(mlp_predict(params, inputs) * targets)
    return log_prior + log_likelihood

def accuracy(params: list, inputs: ndarray, targets: ndarray) -> float:
    target_class: ndarray = np.argmax(targets, axis=1)
    predicted_class: ndarray = np.argmax(mlp_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

if __name__=="__main__": 
    # --- simulation parameters 
    num_samples: int = 128   
    num_bins: int = 64    
    num_examples_per_class: int = 1_000  

    # --- simulate
    print("Generating data...")
    train_inputs, test_inputs, train_labels, test_labels = make_data(n_per_class=num_examples_per_class, n_bins=num_bins, n_draws=num_samples, split=True)

    # --- network architecture 
    layer_sizes: list = [num_bins, 64, 4]
    regularization: float = 1.0

    # --- optimization 
    param_scale: float = 0.1
    batch_size: int = 256
    num_epochs: int = 75
    step_size: float = 0.001

    # --- initialize parameters
    init_params: list = init_mlp_params(param_scale, layer_sizes)

    # --- batching
    num_batches: int = int(np.ceil(len(train_inputs) / batch_size))
    def batch_indices(iteration: int) -> ndarray:
        index: int = iteration % num_batches
        return slice(index * batch_size, (index + 1) * batch_size)

    def objective(params: list, iteration: int) -> float:
        indexes: ndarray = batch_indices(iteration)
        return -mlp_log_posterior(params, train_inputs[indexes], train_labels[indexes], regularization)

    gradient: callable= grad(objective)

    # --- training 
    train_accuracy, test_accuracy = [], []  
    print("     Epoch     |    Train accuracy  |       Test accuracy  ")

    def print_perf(params: list, iteration: int, *args):
        if iteration % num_batches == 0:
            _train_accuracy: float = accuracy(params, train_inputs, train_labels)
            _test_accuracy: float  = accuracy(params, test_inputs, test_labels)
            train_accuracy.append(_train_accuracy)
            test_accuracy.append(_test_accuracy)
            print(f"{(iteration // num_batches):03d}\t|\t{_train_accuracy:0.3f}\t|\t{_test_accuracy:0.3f}")

    optimized_params: list = adam(gradient, init_params, step_size=step_size, num_iters=num_epochs * num_batches, callback=print_perf)

    losses: ndarray = np.vstack((train_accuracy, test_accuracy)).T

    ax = sns.lineplot(data=losses, markers=True)
    ax.set(xlabel='iteration')
    ax.set(ylabel='accuracy')
    ax.set(title='training performance')
    plt.show()
