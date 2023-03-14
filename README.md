# UncertaintyBayesNN
Contains all the code of my bachelor thesis **Uncertainty quantification using bayesian modeling for finite volume neural networks** created at the LS3 in Stuttgart.



## Bayesian Neural Networks

It is expected that you have a basic knowledge of neural networks. Otherwise they are shortly explained in the accompianing thesis.

This bachelor thesis uses stochastic neural networks called BNN(Bayesian neural network). The key difference is that the parameters (weights and biases) have a probability distribution and not just single fixed value. This is motivated by Bayes law, hence the name

$$
    p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}
$$
Where $p(\theta|D)$ is the Posterior, $p(D|\theta)$ is the Likelihood, $p(\theta)$ the prior and $p(D)$ the Evidence.


Two methods:

- Training -- with Variational Inference:
     * Expects some distribution which can be describe by a few parameters e.g. Normal
     * (variational) Parameters can be optimized like in NN 


- Sampling -- with Monte Carlo Markov Chains (MCMC):
     * Evaluates with random weights in a chain
     * Accepts or rejects a proposal to the chain
     * Curse of dimensions make this scale very bad
     * Methods for sampling are:
         * Metropolis Hastings
         * Hamiltonian
         * Crank-Niccolson



## Analysis for best BNN for Retardation Function

* Trains stochasticly with MSE
* Uses the Wasserstein distance to compare actual distribution and distribution of bnn
* Trained architectures:
    *  $$
        [[1, 32, 1],
        [1, 8, 8, 1],
        [1, 4, 9, 4, 1],
        [1, 8, 4, 8, 1]]
       $$
    * Horizontal bayes
      * $$[[0], [0.5], [0.1], [1]]
      $$
    * Vertical bayes
      * $$[[0, 0, 1],
          [0, 0, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 0, 0]]
          $$
    * Special case: Last Layer only
        * $$[[0,-1]]$$
    * Sparse with high uncertainty (rho)


Improvements:




## FINN

* Extends the two sided sorption approach of Gültig
* UQ is done with best BNN from Meta analysis




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
