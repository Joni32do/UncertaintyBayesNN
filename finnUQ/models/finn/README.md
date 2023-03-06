# Testcases
This is a collection of the work Matthias Gültig has done to test the two sided (kinetic and instantanious) diffusion sorption model
$$
\frac{\partial c}{\partial t} = \mathcal{R} \cdot \left( D \Delta c - v \nabla c  - \rho_e \frac{\partial s_k}{\partial t}\right) 
$$ 
$$
\frac{\partial s_k}{\partial t} = \alpha_k \left( (1-f) k_d c^\beta - s_k \right) 
$$
## Training based on synthetic data


### Tests by Gültig
Matthias Gültig did following test, stored in ```archive```

* 27: Learning dummy parameter to validate FINN solution,                 run a<br />
* 28: Learning parameter f,                                               run b<br />
* 29: Learning parameter f, k_d, beta, alpha_k,                           run c<br />
* 30: Learning time dependent functional relation F(c),                   run d<br />
* 31: Learning time dependent functional relations F(c), R(c), G(s_k) with a little amount of epochs<br />
* 32: Learning time dependent functional relations F(c), R(c), G(s_k),    run e<br />


### Initial Tests
Tests from me - Jonathan Schnitzler. Can be deleted

* 50: Some tests
* 51: Some tests
* 52: Test for learning f
* 53: Learning R [1, 100, 100, 1], lr = 0.004, epoch = 2000
* 54: Learn all [1, 100, 100, 1], lr = 0.004, epoch = 10000
* 55: Some test
* 56: R and Parameter (not f), lr = 0.004, epoch = 3000


## Training on Server
Meta training on the architectures:

    "architectures":[[1, 333, 1],
                [1, 98, 98, 1],
                [1, 49, 98, 49, 1],
                [1, 98, 49, 98, 1]],

* 70: Has to be made (I'm a fucking looser and don't understand ```git merge```)

## Training on Personal Computer 

Meta training on the architectures

    "architectures":[[1, 32, 1],
                [1, 8, 8, 1],
                [1, 9, 4, 9, 1],
                [1, 4, 8, 4, 1]],
        
* 80