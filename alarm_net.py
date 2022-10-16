# This file contains an example BN for testing
import sorobn as sbn
import pandas as pd 



# first define network structure
net = sbn.BayesNet(
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
)

# next define conditional probabilities
net.P['Burglary'] = pd.Series({False: 0.999, True: 0.001})
net.P['Earthquake'] = pd.Series({False: 0.998, True: 0.002})
'''net.P['Alarm'] = pd.Series({
    #Alarm, #Burglary, #Earthquake
    (True, True, True): 0.95,
    (True, True, False): 0.94,
    (True, False, True): 0.29,
    (True, False, False): 0.001,

    (False, True, True): 0.05,
    (False, True, False): 0.06,
    (False, False, True): 0.71,
    (False, False, False): 0.999
})

net.P['JohnCalls'] = pd.Series({
    # John, Alarm
    (True, True): 0.90,
    (True, False): 0.05,

    (False, True): 0.10,
    (False, False): 0.95
})

net.P['MaryCalls'] = pd.Series({
    # Mary, Aalarm
    (True, True): 0.70,
    (True, False): 0.01,

    (False, True): 0.30,
    (False, False): 0.99
})'''


net.P['Alarm'] = pd.Series({
    #Alarm, #Burglary, #Earthquake
    (True, True, True): 0.95,
    (True, True, False): 0.05,
    (True, False, True): 0.94,
    (True, False, False): 0.06,

    (False, True, True): 0.29,
    (False, True, False): 0.71,
    (False, False, True): 0.001,
    (False, False, False): 0.999
})

net.P['JohnCalls'] = pd.Series({
    # John, Alarm
    (True, True): 0.90,
    (True, False): 0.1,

    (False, True): 0.05,
    (False, False): 0.95
})

net.P['MaryCalls'] = pd.Series({
    # Mary, Aalarm
    (True, True): 0.70,
    (True, False): 0.3,

    (False, True): 0.01,
    (False, False): 0.99
})


Alarm_Network = net