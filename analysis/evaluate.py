import numpy as np
from scipy.stats import wasserstein_distance


def eval_Bayes_net(net, x_eval, samples, quantile = 0.025):
    '''
    Evaluates the input {x_eval} for {samples} times
    (bars, samples)
    '''  
    bars = x_eval.size(dim = 0)
    y_preds = np.zeros((bars, samples))
    for i in range(samples):
        y_preds[:,i] = net.forward(x_eval).detach().numpy().flatten()
         
    # Calculate mean and quantiles
    mean = np.mean(y_preds, axis=1)
    lower = np.quantile(y_preds, quantile, axis=1)
    upper = np.quantile(y_preds, 1-quantile, axis=1)
    return y_preds,mean,lower,upper


def calc_water(y_preds, y_eval, evaluation_type = "mean"):
    '''
    calculates for each bar (single 1D coordinate of input space) with samples the wasserstein_distance 
    
    return
        -average of wasserstein over all bars
    
    '''
    assert np.shape(y_preds) == np.shape(y_eval)
    
    bars, samples = np.shape(y_preds)
    water = np.zeros(bars)
    for i in range(bars):
        water[i] = wasserstein_distance(y_preds[i,:], y_eval[i,:])
    
    if evaluation_type is "mean":
        return np.mean(water)
    elif evaluation_type is "mat":
        return water
    else:
        raise NotImplementedError("This return type was not implemented for Wasserstein")


