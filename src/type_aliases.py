import argparse 

import os 
import pathlib 
from typing import *

import autograd
import numpy

# --- optional imports 
try: 
    import tensorflow as tf 
    tensor: type = tf.Tensor
except ModuleNotFoundError: 
    pass

# --- aggregate types for numerics 
ndarray: type = Union[numpy.ndarray, autograd.numpy.ndarray]

# --- string types 
path_t: type = Union[os.PathLike, pathlib.Path, str]

# --- namespaces 
namespace: type = argparse.Namespace
