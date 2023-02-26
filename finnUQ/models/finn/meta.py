'''
Meta analysis of Bayesian neural networks in FINN

        Task: Two sided sorption (instant and kinetic)

        Params:
            See params.json
        Constants:
            See config.json

    What part is trainable 
        o Retardation Factor
    


'''


architectures = [[1 ,32, 1],
                 [1, 8, 8, 1],
                 [1, 9, 4, 9, 1],
                 [1, 8, 4, 8, 1]]


#Other Options below 
bayes_arc     = [[0, 0, 0, 1], 
                 [0, 0, 8, 0], 
                 [0, 0, 9, 0, 0], 
                 [0, 0, 4, 0, 0]]


testcases = [architectures, bayes_arc]


#TODO: Alter config file according to options










"""



bayes_arc

bayes factor





"""