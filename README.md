# UncertaintyBayesNN
Contains all the code of my bachelor thesis **Uncertainty quantification using bayesian modeling for finite volume neural networks** created at the LS3 in Stuttgart.

## Introduction into Uncertainty

Perhabs I will firstfully explore how to meassure uncertainty in a dataset and introduce some common meassures from statistics. This would be a good opportunity to explain:


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
```
@ARTICLE{9756596,
author={Jospin, Laurent Valentin and Laga, Hamid and Boussaid, Farid and Buntine, Wray and Bennamoun, Mohammed},
journal={IEEE Computational Intelligence Magazine}, 
title={Hands-On Bayesian Neural Networks—A Tutorial for Deep Learning Users}, 
year={2022},
volume={17},
number={2},
pages={29-48},
doi={10.1109/MCI.2022.3155327}
}

@misc{https://doi.org/10.48550/arxiv.2104.06010,
  doi = {10.48550/ARXIV.2104.06010},
  url = {https://arxiv.org/abs/2104.06010},
  author = {Praditia, Timothy and Karlbauer, Matthias and Otte, Sebastian and Oladyshkin, Sergey and Butz, Martin V. and Nowak, Wolfgang},
  keywords = {Machine Learning (cs.LG), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Finite Volume Neural Network: Modeling Subsurface Contaminant Transport},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}

@misc{https://doi.org/10.48550/arxiv.2111.11798,
  doi = {10.48550/ARXIV.2111.11798},
  url = {https://arxiv.org/abs/2111.11798},
  author = {Karlbauer, Matthias and Praditia, Timothy and Otte, Sebastian and Oladyshkin, Sergey and Nowak, Wolfgang and Butz, Martin V.},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Composing Partial Differential Equations with Physics-Aware Neural Networks},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
