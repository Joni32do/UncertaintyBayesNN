# Includes trained models. Training based on synthetic data

27: Learning dummy parameter to validate FINN solution,                 run a<br />
28: Learning parameter f,                                               run b<br />
29: Learning parameter f, k_d, beta, alpha_k,                           run c<br />
30: Learning time dependent functional relation F(c),                   run d<br />
31: Learning time dependent functional relations F(c), R(c), G(s_k) with a little amount of epochs<br />
32: Learning time dependent functional relations F(c), R(c), G(s_k),    run e<br />

50: Some tests
51: Some tests
52: Test for learning f
53: Learning R [1, 100, 100, 1], lr = 0.004, epoch = 2000
54: Learn all [1, 100, 100, 1], lr = 0.004, epoch = 10000
55: Some test
56: R and Parameter (not f), lr = 0.004, epoch = 3000



70: Training on Server

80: Training on PC 
[2,8,8,1]
[0,0,4,1]
80: Pre1000
81: Pre1000Epoch1000 
#Warum sagt es, es wären 637 Parameter zum trainieren - warum ist es jetzt wieder nan?



        "architectures":[[1, 333, 1],
                        [1, 98, 98, 1],
                        [1, 49, 98, 49, 1],
                        [1, 98, 49, 98, 1]],