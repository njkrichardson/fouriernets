import numpy as np 
import pandas as pd
import numpy.random as npr
import seaborn as sns 
import matplotlib.pyplot as plt
from math import pi
from utils import make_data, plot_sample, train_test_split, one_hot_decoding, plot_distribution
from distributions import uniform, linear, semi_circular, von_mises
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
from nnet import init_mlp_params, mlp_log_posterior
sns.set_style('white')
sns.set(rc={'figure.figsize':(18, 11)})
npr.seed(0)

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

if __name__=="__main__": 
    # simulation parameters 
    n_draws = 128   # number of draws from each distribution 
    n_bins = 64      # number of bins in each distribution 

    # network parameters 
    layer_sizes = [n_bins, 64, 4]
    L2_reg = 1.0

    # training parameters
    param_scale = 0.1
    batch_size = 256
    num_epochs = 75
    step_size = 0.001

    # data size 
    n_data = 1000  # number of data per class

    # generate data 
    print("Generating data...")
    train_inputs, test_inputs, train_labels, test_labels = make_data(n_per_class=n_data, n_bins=n_bins, n_draws=n_draws, split=True)

    # initialize the net
    init_params = init_mlp_params(param_scale, layer_sizes)

    # batching parameters
    num_batches = int(np.ceil(len(train_inputs) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # define training objective: negative log marginal likelihood 
    def objective(params, iter):
        idx = batch_indices(iter)
        return -mlp_log_posterior(params, train_inputs[idx], train_labels[idx], L2_reg)

    # get gradient of objective using autograd
    objective_grad = grad(objective)


    # log training and test loss 
    train_accs, test_accs = [], []  
    print("     Epoch     |    Train accuracy  |       Test accuracy  ")

    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_inputs, train_labels)
            test_acc  = accuracy(params, test_inputs, test_labels)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)

    losses = pd.DataFrame(data=np.vstack((train_accs, test_accs)).T, columns=['train_acc', 'test_acc'])
    ax = sns.lineplot(data=losses, markers=True)
    ax.set(xlabel='iteration')
    ax.set(ylabel='accuracy')
    ax.set(title='training performance')
    plt.show()