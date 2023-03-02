import numpy as np
import argparse


def n_params_architectures(arcs):
    '''
    Calculates number of parameters (weights and biases) of an architecture
    '''
    n = np.zeros(len(arcs))
    for i, arc in enumerate(arcs):
        n_w = 0
        for j in range(len(arc)-1):
            n_w += arc[j] * arc[j+1]
        n_b = sum(arc) - arc[0]
        n[i] = n_w + n_b
    return n

def n_params_architecture(arc):
    '''
    Calculates number of parameters (weights and biases) of an architecture
    '''
    n_w = 0
    for j in range(len(arc)-1):
        n_w += arc[j] * arc[j+1]
    n_b = sum(arc) - arc[0]
    return n_w + n_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", nargs='+', type = int, 
                        help="input the architecture as an iterable which you want to be calculated")

    args = parser.parse_args()
    arc = args.architecture
    print(f"Networks architecture {arc}")
    n_params = n_params_architecture(args.architecture)
    print(f"\t has {n_params} parameter")