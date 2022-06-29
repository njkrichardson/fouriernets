import numpy as np 
import numpy.random as npr
from distributions import uniform, linear, semi_circular, von_mises
from builtins import range
import numpy as np
import pandas as pd
import sys, time
import matplotlib.pyplot as plt 
import seaborn as sns 
from math import pi 
sns.set_style('white')
COLORS = np.array([
[106,61,154],  # Dark colors
[31,120,180],
[51,160,44],
[227,26,28],
[255,127,0],
[166,206,227],  # Light colors
[178,223,138],
[251,154,153],
[253,191,111],
[202,178,214],
]) / 256.0

def plot_sample(inputs : np.ndarray, targets : np.ndarray, n_samples : int = 4): 
    classes = {0: 'semicircular', 1:'uniform', 2:'linear', 3:'von mises'}
    n = inputs.shape[0]
    targets = one_hot_decoding(targets)
    
    dist_by_type = [np.where(targets==0)[0], np.where(targets==1)[0], np.where(targets==2)[0], np.where(targets==3)[0]]
    j = 0 
            
    fig, axs = plt.subplots(1, 4, figsize=(15, 6), facecolor='w', edgecolor='k', subplot_kw={'projection': 'polar'})
    plt.suptitle('samples')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()

    for i in range(n_samples):
        idx = dist_by_type[j][i]
        name=targets[idx]
        radius, theta = unroll_distribution(inputs[idx])
        axs[i].scatter(theta, radius, 5, c=COLORS[j].reshape(-1, 1).T)
        axs[i].set_title(classes[name], pad=15)
        j+=1 
        if j > 3: 
            j = 0 
    plt.show()

round = (lambda x: lambda y: int(x(y)))(round)

# NOTE: datetime.timedelta.__str__ doesn't allow formatting the number of digits
def sec2str(seconds):
    hours, rem = divmod(seconds,3600)
    minutes, seconds = divmod(rem,60)
    if hours > 0:
        return '%02d:%02d:%02d' % (hours,minutes,round(seconds))
    elif minutes > 0:
        return '%02d:%02d' % (minutes,round(seconds))
    else:
        return '%0.2f' % seconds

def progprint_xrange(*args,**kwargs):
    xr = range(*args)
    return progprint(xr,total=len(xr),**kwargs)

def progprint(iterator,total=None,perline=25,show_times=True):
    times = []
    idx = 0
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (idx+1) % perline == 0:
            if show_times:
                avgtime = np.mean(times)
                if total is not None:
                    eta = sec2str(avgtime*(total-(idx+1)))
                    sys.stdout.write((
                        '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n'
                                % (numdigits,numdigits)) % (idx+1,total,avgtime,eta))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times),total))

def make_data(n_per_class : int = 100, n_bins : int = 12, n_draws : int = 100, split : bool = False, test_proportion : float = 0.2, \
              kappa : float = 0.5, slope_mag : float = 5., bias : float = .85): 
    n_data = n_per_class * 4
    inputs, targets = np.zeros((n_data, n_bins)), np.zeros((n_data, 4))
    
    for idx in progprint_xrange(0, n_data-3, 4): 
        # generate sample data
        inputs[idx], targets[idx][0] = semi_circular(n_bins, n_draws, bias=bias), 1 
        inputs[idx+1], targets[idx+1][1] = uniform(n_bins, n_draws), 1
        inputs[idx+2], targets[idx+2][2]= linear(n_bins, n_draws, slope_mag=slope_mag), 1
        inputs[idx+3], targets[idx+3][3] = von_mises(n_bins, n_draws, kappa=kappa), 1
    
    if split is True: 
        return train_test_split(inputs, targets, test_proportion=test_proportion)

    return inputs, targets

def train_test_split(inputs, targets, test_proportion : float = 0.2): 
    n_data = inputs.shape[0]
    mask = npr.choice(n_data, size=int(n_data*(1-test_proportion)), replace=False)
    train_inputs, train_targets = inputs[(mask)], targets[(mask)]
    test_inputs, test_targets = np.delete(inputs, mask, 0), np.delete(targets, mask, 0)
    return train_inputs, test_inputs, train_targets, test_targets

def one_hot_decoding(oh_arr : np.ndarray) -> np.ndarray: 
    return np.array([np.where(oh_arr[i]==1)[0][0] for i in range(len(oh_arr))])

def plot_distribution(distribution : np.ndarray, r : int = 5, name : str = 'distribution'): 
    radius, theta = unroll_distribution(distribution, r=r)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(theta, radius)
    ax.title.set_text(name)
    plt.show()
    
    
def unroll_distribution(distribution : np.ndarray, r : int = 5): 
    radius, theta = [], [] 

    for i in range(len(distribution)): 
        angle = (i / len(distribution)) * (pi * 2)
        for j in range(int(distribution[i])): 
            radius.append(0.1*j+r)
            theta.append(angle)

    return radius, theta

def load_maddie(): 
    # import Maddie's data to confirm distribution discrimination 
    maddie_sc = pd.read_csv('/Users/nickrichardson/Desktop/academics/2019-20/fouriernets/maddie/SemiCircleData_testing.csv', header=None).values[:, :-1]
    maddie_linear = pd.read_csv('/Users/nickrichardson/Desktop/academics/2019-20/fouriernets/maddie/LineData_testing.csv', header=None).values[:, :-1]
    maddie_uniform = pd.read_csv('/Users/nickrichardson/Desktop/academics/2019-20/fouriernets/maddie/UniformData_testing.csv', header=None).values[:, :-1]
    maddie_vonmises = pd.read_csv('/Users/nickrichardson/Desktop/academics/2019-20/fouriernets/maddie/VonMisesData_testing.csv', header=None).values[:, :-1]
    
    n_p_class = maddie_linear.shape[0]
    inputs = np.vstack((maddie_linear, maddie_sc, maddie_uniform, maddie_vonmises))
    targets = np.vstack((np.tile([1, 0, 0, 0], n_p_class).reshape(n_p_class, 4), np.tile([0, 1, 0, 0], n_p_class).reshape(n_p_class, 4),\
                     np.tile([0, 0, 1, 0], n_p_class).reshape(n_p_class, 4), np.tile([0, 0, 0, 1], n_p_class).reshape(n_p_class, 4)))
    return train_test_split(inputs, targets)

#     df = pd.DataFrame({'radius': radius, name : theta})

#     # Convert the dataframe to long-form or "tidy" format
#     df = pd.melt(df, id_vars=['radius'], var_name='distribution type', value_name='theta')

#     # Set up a grid of axes with a polar projection
#     g = sns.FacetGrid(df, col='distribution type', hue="distribution type",
#                       subplot_kws=dict(projection='polar'), height=4.5,
#                       sharex=False, sharey=False, despine=False, margin_titles=False)

#     # Draw a scatterplot
#     g.map(sns.scatterplot, "theta", "radius");