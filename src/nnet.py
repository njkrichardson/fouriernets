import numpy as np 
import numpy.random as npr 
from autograd.scipy.special import logsumexp
from autograd.misc.flatten import flatten
import pandas as pd 
import seaborn as sns
from autograd import grad
from autograd.misc.optimizers import adam


# basic mlp 

def init_mlp_params(scale, layer_sizes, rs = npr.RandomState(0)):
    """Build a list of (weights, biases) tuples.
    
    Parameters
    ----------
    scale : float
        parameter scale factor
    layer_sizes : list of ints 
        list of network layer sizes
    rs : [], optional
        uses a consistent seed, by default npr.RandomState(0)
    
    Returns
    -------
    list 
        mlp params, a list of (weights, biases) tuples 
    """
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# simple wrapping/training utils 

def mlp_predict(params, inputs):
    """Instantiates a neural network (currently for classification).
    
    Parameters
    ----------
    params : list of (weights, bias) tuples
        params of the net 
    inputs : np.ndarray
        input data assumed to be in (samples x features) form
    
    Returns
    -------
    np.ndarray
        normalized class log-probabilities
    """
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def l2_norm(params):
    """Computes squared l2 norm of params. 
    
    Parameters
    ----------
    params : list of (weights, biases) tuples
        parameters of the net
    
    Returns
    -------
    float
        squared euclidean norm of params
    """
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def mlp_log_posterior(params, inputs, targets, L2_reg):
    """Compute the log posterior probability of the mlp. 
    
    Parameters
    ----------
    params : list of (weights, biases) tuples
        parameters of the net
    inputs : np.ndarray
        inputs to use to evaluate the likelihood  
    targets : np.ndarray
        targets to use to evaluate the likelihood 
    L2_reg : float 
        corresponds to a metaparameter for the prior weight size
    
    Returns
    -------
    float
        log posterior probability 
    """
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(mlp_predict(params, inputs) * targets)
    return log_prior + log_lik

