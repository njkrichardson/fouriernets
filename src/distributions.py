import numpy as np 
import numpy.random as npr 
from math import pi, exp, cos
import pandas as pd
import seaborn as sns

def uniform(num_bins : int, num_points : int, visualize : bool = False) -> np.ndarray: 
    '''
    Generates data from a uniform distribution over the unit circle. 
    '''
    result = np.zeros(num_bins)
    
    for _ in range(num_points):
        result[npr.randint(0, num_bins)] += 1
        
    if visualize is True: 
        plot_distribution(result, 'uniform')
        
    return result 

def linear(num_bins : int, num_points : int, slope_mag : int = 5, visualize : bool = False) -> np.ndarray: 
    '''
    Generates linearly distributed data on the unit circle. 
    '''
    probabilities, elements = np.zeros(num_bins), np.arange(num_bins)
    result = np.zeros(num_bins) 
    
    # define slope parameters
    slope = (npr.random() + slope_mag) / 10 
    e = (-slope / 2) * (num_bins ** 2)
    
    for i in range(num_bins): 
        probabilities[i] = (-slope/(2 * e))*(((i+1)**2) - (i)**2)
        
    probabilities = list(probabilities)
    
    for _ in range(num_points): 
        sample = int(npr.choice(elements, 1, probabilities))
        result[sample] += 1
        
    # induce a circular shift
    result = np.roll(result, npr.randint(0, num_bins))
        
    if visualize is True: 
        plot_distribution(result, 'linear')

    return result

def von_mises(num_bins : int, num_points : int, kappa : float = 0.5, mean_loc : float = 0.,visualize : bool = False) -> np.ndarray: 
    '''
    Generates von mises distributed data on the unit circle. 
    '''
    samples = npr.vonmises(mean_loc, kappa=kappa, size=num_points)
    result = np.histogram(samples, bins=num_bins, range=(-pi, pi))[0]
    
    # induce a random circular shift
    result = np.roll(result, npr.randint(0, num_bins))
    
    if visualize is True: 
        plot_distribution(result, 'Von Mises')
        
    return result 

def semi_circular(num_bins : int, num_points : int, bias : float = .70, visualize : bool = False) -> np.ndarray: 
    '''
    Generates semicircular distributed data on the unit circle. 
    '''
    assert num_bins % 2 == 0
    p = int(num_bins/2)
    
    dominant_half, submissive_half = np.zeros(p), np.zeros(p)
    
    for _ in range(num_points):
        if npr.random() < bias:
            dominant_half[npr.randint(0, p, dtype='int')] += 1
        else: 
            submissive_half[npr.randint(0, p, dtype='int')] += 1
    
    result = np.hstack((dominant_half, submissive_half))
    
    # induce a random circular shift
    result = np.roll(result, npr.randint(0, num_bins))
    
    if visualize is True: 
        plot_distribution(result, 'semi_circular')
        
    return result 
