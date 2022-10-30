# UncertaintyBayesNN
Contains all the code of my bachelor thesis **Uncertainty quantification using bayesian modeling for finite volume neural networks** created at the LS3 in Stuttgart.

## Introduction into Uncertainty

Perhabs I will firstfully explore how to meassure uncertainty in a dataset and introduce some common meassures from statistics. This would be a good opportunity to explain:

* Confusion matrix
* Sensitivity and Specificity
* Entropy
* ROC (Receiver Operator Characteistic) and Calibration Curve (Check if indeed they represent the same dynamic)
* AOC (Area under the curve)


## Introduction into Neural Networks

A small introduction into NN which I will use to give a brief insight in:

* Architecture
* Data-Set (maybe DataLoader,...)
* Hyperparameter tuning:
  * Learning rate (+ schedule)
  * Optimizer
  * Drop-Out
  * Gamma
  * Early stopping
* Loss
* Regularization
* Ensemble Learning (which I will use as a bridge to motivate BNN)


## Bayesian Neural Networks

Probability distribution on parameters (or activation functions which can be choosen such that they are equivilant)

## TODO:
- Varying Architecture
- More difficult functions (R^n->R^m, PDEs)

Training

- Variational Inference:
     * Different models of BayesLayer
     * Training algorithms
- MCMC:
     * Metropolis Hastings
     * Hamiltonian
     * NUTS

Inference

- Performance testing
     * Calibration curve
     * AUC and Abbreviation

## FINN

I have to discuss this with Nils -> Many methods are already implemented



## Dependencies

- PyTorch 1.12.1
- Numpy 1.20.3

## Sources


