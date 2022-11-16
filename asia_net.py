# This file contains an example BN for testing
import sorobn as sbn
import pandas as pd 

# first define network structure
"""net = sbn.BayesNet(
    ('Smoke', 'Lung'),
    ('Asia', 'Tub'),
    ('Lung', 'Either'),
    ('Tub', 'Either'),
    ('Either', 'Xray'),
    ('Smoke', 'Bronc'),
    ('Bronc', 'Dysp'),
    ('Either', 'Dysp'),
)"""

net = sbn.BayesNet(
    ('Asia', 'Tub'),
    ('Smoke', ['Lung', 'Bronc']),
    (['Tub', 'Lung'], 'Either'),
    ('Either', ['Xray', 'Dysp']),
    ('Bronc', 'Dysp')
)


# next define conditional probabilities
net.P['Asia'] = pd.Series({False: 0.99, True: 0.01})
net.P['Smoke'] = pd.Series({False: 0.5, True: 0.5})


net.P['Tub'] = pd.Series({
    #(Asia, Tub)
    (True, True): 0.05,
    (True, False): 0.95,
    (False, True): 0.01,
    (False, False): 0.99
})

net.P['Lung'] = pd.Series({
    #(Smoke, Lung)
    (True, True): 0.1,
    (True, False): 0.9,
    (False, True): 0.01,
    (False, False): 0.99
})

net.P['Bronc'] = pd.Series({
    #(Smoke, Bronc)
    (True, True): 0.6,
    (True, False): 0.4,
    (False, True): 0.3,
    (False, False): 0.7
})

net.P['Xray'] = pd.Series({
    #(Either, Xray)
    (True, True): 0.98,
    (True, False): 0.02,
    (False, True): 0.05,
    (False, False): 0.95
})


net.P['Either'] = pd.Series({
    #(Lung, Tub, Either)
    (True, True, True): 1.0,
    (True, True, False): 0.0,
    (True, False, True): 1.0,
    (True, False, False): 0.0,

    (False, True, True): 1.0,
    (False, True, False): 0.0,
    (False, False, True): 0.0,
    (False, False, False): 1.0,
})

net.P['Dysp'] = pd.Series({
    #(Bronc, Either, Dysp)
    (True, True, True): 0.9,
    (True, True, False): 0.1,
    (True, False, True): 0.8,
    (True, False, False): 0.2,

    (False, True, True): 0.7,
    (False, True, False): 0.3,
    (False, False, True): 0.1,
    (False, False, False): 0.9,
})




Asia_Network = net