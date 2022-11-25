# This file contains an example BN for testing
import sorobn as sbn
import pandas as pd 

# first define network structure
net = sbn.BayesNet(
    ('Asia', 'Tub'),
    ('Smoke', ['Lung', 'Bronc']),
    (['Tub', 'Lung'], 'Either'),
    ('Either', ['Xray', 'Dysp']),
    ('Bronc', 'Dysp')
)


# next define conditional probabilities
net.P['Asia'] = pd.Series({'no': 0.99, 'yes': 0.01})
net.P['Smoke'] = pd.Series({'no': 0.5, 'yes': 0.5})


net.P['Tub'] = pd.Series({
    #(Asia, Tub)
    ('yes', 'yes'): 0.05,
    ('yes', 'no'): 0.95,
    ('no', 'yes'): 0.01,
    ('no', 'no'): 0.99
})

net.P['Lung'] = pd.Series({
    #(Smoke, Lung)
    ('yes', 'yes'): 0.1,
    ('yes', 'no'): 0.9,
    ('no', 'yes'): 0.01,
    ('no', 'no'): 0.99
})

net.P['Bronc'] = pd.Series({
    #(Smoke, Bronc)
    ('yes', 'yes'): 0.6,
    ('yes', 'no'): 0.4,
    ('no', 'yes'): 0.3,
    ('no', 'no'): 0.7
})

net.P['Xray'] = pd.Series({
    #(Either, Xray)
    ('yes', 'yes'): 0.98,
    ('yes', 'no'): 0.02,
    ('no', 'yes'): 0.05,
    ('no', 'no'): 0.95
})


net.P['Either'] = pd.Series({
    #(Lung, Tub, Either)
    ('yes', 'yes', 'yes'): 1.0,
    ('yes', 'yes', 'no'): 0.0,
    ('yes', 'no', 'yes'): 1.0,
    ('yes', 'no', 'no'): 0.0,

    ('no', 'yes', 'yes'): 1.0,
    ('no', 'yes', 'no'): 0.0,
    ('no', 'no', 'yes'): 0.0,
    ('no', 'no', 'no'): 1.0,
})

net.P['Dysp'] = pd.Series({
    #(Bronc, Either, Dysp)
    ('yes', 'yes', 'yes'): 0.9,
    ('yes', 'yes', 'no'): 0.1,
    ('yes', 'no', 'yes'): 0.8,
    ('yes', 'no', 'no'): 0.2,

    ('no', 'yes', 'yes'): 0.7,
    ('no', 'yes', 'no'): 0.3,
    ('no', 'no', 'yes'): 0.1,
    ('no', 'no', 'no'): 0.9,
})




Asia_Network = net