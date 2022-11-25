# This file contains an example BN for testing
import sorobn as sbn
import pandas as pd 

# first define network structure
net = sbn.BayesNet(
    ('Age', ['SocioEcon', 'RiskAversion', 'GoodStudent', 'SeniorTrain', 'DrivingSkill', 'MedCost']),
    ('SocioEcon', ['GoodStudent', 'RiskAversion','MakeModel', 'OtherCar', 'VehicleYear', 'HomeBase','AntiTheft']),
    ('RiskAversion', ['SeniorTrain', 'DrivHist', 'DrivQuality', 'MakeModel', 'VehicleYear','HomeBase', 'AntiTheft']),
    ('SeniorTrain', 'DrivingSkill'),
    ('HomeBase','Theft'),
    ('AntiTheft','Theft'),
    ('DrivingSkill',['DrivHist','DrivQuality']),
    ('DrivQuality','Accident'),
    ('MakeModel',['Airbag','Antilock','RuggedAuto','CarValue']),
    ('VehicleYear',['Airbag','Antilock','RuggedAuto','CarValue']),
    ('Mileage',['CarValue','Accident']),
    ('Airbag','Cushioning'),
    ('Antilock','Accident'),
    ('RuggedAuto',['Cushioning','OtherCarCost','ThisCarDam']),
    ('CarValue',['ThisCarCost','Theft']),

    ('Cushioning','MedCost'),
    ('Accident',['MedCost','ILiCost','OtherCarCost', 'ThisCarDam']),
    ('Theft','ThisCarCost'),
    ('OtherCarCost','PropCost'),
    ('ThisCarDam','ThisCarCost'),
    ('ThisCarCost','PropCost'),

)


# next define conditional probabilities
net.P['Age'] = pd.Series({'Adolescent': 0.2, 'Adult': 0.6, 'Senior': 0.2})
net.P['SocioEcon'] = pd.Series({
    #(Age, SE)
    ('Adolescent', 'Prole'): 0.4,
    ('Adolescent', 'Middle'): 0.4,
    ('Adolescent', 'UpperMiddle'): 0.19,
    ('Adolescent', 'Wealthy'): 0.01,
    ('Adult', 'Prole'): 0.4,
    ('Adult', 'Middle'): 0.4,
    ('Adult', 'UpperMiddle'): 0.19,
    ('Adult', 'Wealthy'): 0.01,
    ('Senior', 'Prole'): 0.5,
    ('Senior', 'Middle'): 0.2,
    ('Senior', 'UpperMiddle'): 0.29,
    ('Senior', 'Wealthy'): 0.01,
})

net.P['RiskAversion'] = pd.Series({
    #(Age, SE, RA)
    ('Adolescent', 'Prole', 'Psychopath'): 0.02,
    ('Adolescent', 'Prole', 'Adventurous'): 0.58,
    ('Adolescent', 'Prole', 'Normal'): 0.3,
    ('Adolescent', 'Prole', 'Cautious'): 0.1, 
    ('Adolescent', 'Middle','Psychopath'): 0.02,
    ('Adolescent', 'Middle','Adventurous'): 0.38,
    ('Adolescent', 'Middle','Normal'): 0.5,
    ('Adolescent', 'Middle','Cautious'): 0.1, 
    ('Adolescent', 'UpperMiddle','Psychopatah'): 0.02,
    ('Adolescent', 'UpperMiddle','Adventurous'): 0.48,
    ('Adolescent', 'UpperMiddle','Normal'): 0.40,
    ('Adolescent', 'UpperMiddle','Cautious'): 0.10,
    ('Adolescent', 'Wealthy', 'Psychopath'): 0.02,
    ('Adolescent', 'Wealthy', 'Adventurous'): 0.58,
    ('Adolescent', 'Wealthy', 'Normal'): 0.3,
    ('Adolescent', 'Wealthy', 'Cautious'): 0.1,

    ('Adult', 'Prole','Psychopath'): 0.015,
    ('Adult', 'Prole','Adventurous'): 0.285,
    ('Adult', 'Prole','Normal'): 0.5,
    ('Adult', 'Prole','Cautious'): 0.2, 
    ('Adult', 'Middle','Psychopath'): 0.015,
    ('Adult', 'Middle','Adventurous'): 0.185,
    ('Adult', 'Middle','Normal'): 0.6,
    ('Adult', 'Middle','Cautious'): 0.2,
    ('Adult', 'UpperMiddle','Psychopatah'): 0.015,
    ('Adult', 'UpperMiddle','Adventurous'): 0.285,
    ('Adult', 'UpperMiddle','Normal'): 0.50,
    ('Adult', 'UpperMiddle','Cautious'): 0.20,
    ('Adult', 'Wealthy', 'Psychopath'): 0.015,
    ('Adult', 'Wealthy', 'Adventurous'): 0.285,
    ('Adult', 'Wealthy', 'Normal'): 0.4,
    ('Adult', 'Wealthy', 'Cautious'): 0.3,

    ('Senior', 'Prole', 'Psychopath'): 0.01,
    ('Senior', 'Prole', 'Adventurous'): 0.09,
    ('Senior', 'Prole', 'Normal'): 0.4,
    ('Senior', 'Prole', 'Cautious'): 0.5, #good
    ('Senior', 'Middle','Psychopath'): 0.01,
    ('Senior', 'Middle','Adventurous'): 0.04,
    ('Senior', 'Middle','Normal'): 0.35,
    ('Senior', 'Middle','Cautious'): 0.60,
    ('Senior', 'UpperMiddle','Psychopatah'): 0.01,
    ('Senior', 'UpperMiddle','Adventurous'): 0.09,
    ('Senior', 'UpperMiddle','Normal'): 0.40,
    ('Senior', 'UpperMiddle','Cautious'): 0.50,
    ('Senior', 'Wealthy', 'Psychopath'): 0.01,
    ('Senior', 'Wealthy', 'Adventurous'): 0.09,
    ('Senior', 'Wealthy', 'Normal'): 0.4,
    ('Senior', 'Wealthy', 'Cautious'): 0.5,
})

net.P['SeniorTrain'] = pd.Series({
    # Age, RA, ST
    ('Adolescent','Psychopath','True'): 0.0,
    ('Adolescent','Psychopath','False'): 1.0,
    ('Adult','Psychopath','True'): 0.0,
    ('Adult','Psychopath','False'): 1.0,
    ('Senior','Psychopath','True'): 0.000001,
    ('Senior','Psychopath','False'): 0.999999,

    ('Adolescent','Adventurous','True'): 0.0,
    ('Adolescent','Adventurous','False'): 1.0,
    ('Adult','Adventurous','True'): 0.0,
    ('Adult','Adventurous','False'): 1.0,
    ('Senior','Adventurous','True'): 0.000001,
    ('Senior','Adventurous','False'): 0.999999,


    ('Adolescent','Normal','True'): 0.0,
    ('Adolescent','Normal','False'): 1.0,
    ('Adult','Normal','True'): 0.0,
    ('Adult','Normal','False'): 1.0,
    ('Senior','Normal','True'): 0.3,
    ('Senior','Normal','False'): 0.7,

    ('Adolescent','Cautious','True'): 0.0,
    ('Adolescent','Cautious','False'): 1.0,
    ('Adult','Cautious','True'): 0.0,
    ('Adult','Cautious','False'): 1.0,
    ('Senior','Cautious','True'): 0.9,
    ('Senior','Cautious','False'): 0.1,
})

net.P['HomeBase'] = pd.Series({
    # RA, SE, HB
    ('Psychopath', 'Prole', 'Secure') = 0.000001,
    ('Psychopath', 'Prole', 'City') = 0.8,
    ('Psychopath', 'Prole', 'Suburb') = 0.049999,
    ('Psychopath', 'Prole', 'Rural') = 0.15,

    ('Adventurous', 'Prole', 'Secure') = 0.000001,
    ('Adventurous', 'Prole', 'City') = 0.8,
    ('Adventurous', 'Prole', 'Suburb') = 0.05,
    ('Adventurous', 'Prole', 'Rural') = 0.149999,

    ('Normal', 'Prole', 'Secure') = 0.000001,
    ('Normal', 'Prole', 'City') = 0.8,
    ('Normal', 'Prole', 'Suburb') = 0.05,
    ('Normal', 'Prole', 'Rural') = 0.149999,

    ('Cautious', 'Prole', 'Secure') = 0.000001,
    ('Cautious', 'Prole', 'City') = 0.8,
    ('Cautious', 'Prole', 'Suburb') = 0.05,
    ('Cautious', 'Prole', 'Rural') = 0.149999,

    ('Psychopath', 'Middle', 'Secure') = 0.15,
    ('Psychopath', 'Middle', 'City') = 0.8,
    ('Psychopath', 'Middle', 'Suburb') = 0.04,
    ('Psychopath', 'Middle', 'Rural') = 0.01,

    ('Adventurous', 'Middle', 'Secure') = 0.01,
    ('Adventurous', 'Middle', 'City') = 0.25,
    ('Adventurous', 'Middle', 'Suburb') = 0.6,
    ('Adventurous', 'Middle', 'Rural') = 0.14,

    ('Normal', 'Middle', 'Secure') = 0.299999,
    ('Normal', 'Middle', 'City') = 0.000001,
    ('Normal', 'Middle', 'Suburb') = 0.6,
    ('Normal', 'Middle', 'Rural') = 0.1,

    ('Cautious', 'Middle', 'Secure') = 0.95,
    ('Cautious', 'Middle', 'City') = 0.000001,
    ('Cautious', 'Middle', 'Suburb') = 0.024445,
    ('Cautious', 'Middle', 'Rural') = 0.025554,

    
    ('Psychopath', 'UpperMiddle', 'Secure') = 0.35,
    ('Psychopath', 'UpperMiddle', 'City') = 0.6,
    ('Psychopath', 'UpperMiddle', 'Suburb') = 0.04,
    ('Psychopath', 'UpperMiddle', 'Rural') = 0.01,

    ('Adventurous', 'UpperMiddle', 'Secure') = 0.2,
    ('Adventurous', 'UpperMiddle', 'City') = 0.4,
    ('Adventurous', 'UpperMiddle', 'Suburb') = 0.3,
    ('Adventurous', 'UpperMiddle', 'Rural') = 0.1,

    ('Normal', 'UpperMiddle', 'Secure') = 0.5,
    ('Normal', 'UpperMiddle', 'City') = 0.000001,
    ('Normal', 'UpperMiddle', 'Suburb') = 0.4,
    ('Normal', 'UpperMiddle', 'Rural') = 0.099999,

    ('Cautious', 'UpperMiddle', 'Secure') = 0.999997,
    ('Cautious', 'UpperMiddle', 'City') = 0.000001,
    ('Cautious', 'UpperMiddle', 'Suburb') = 0.000001,
    ('Cautious', 'UpperMiddle', 'Rural') = 0.000001,

    ('Psychopath', 'Wealthy', 'Secure') = 0.489999,
    ('Psychopath', 'Wealthy', 'City') = 0.5,
    ('Psychopath', 'Wealthy', 'Suburb') = 0.000001,
    ('Psychopath', 'Wealthy', 'Rural') = 0.010000,

    ('Adventurous', 'Wealthy', 'Secure') = 0.950000,
    ('Adventurous', 'Wealthy', 'City') = 0.000001,
    ('Adventurous', 'Wealthy', 'Suburb') = 0.000001,
    ('Adventurous', 'Wealthy', 'Rural') = 0.489999,

    ('Normal', 'Wealthy', 'Secure') = 0.850000,
    ('Normal', 'Wealthy', 'City') = 0.000001,
    ('Normal', 'Wealthy', 'Suburb') = 0.001000,
    ('Normal', 'Wealthy', 'Rural') = 0.148999,

    ('Cautious', 'Wealthy', 'Secure') = 0.999997,
    ('Cautious', 'Wealthy', 'City') = 0.000001,
    ('Cautious', 'Wealthy', 'Suburb') = 0.000001,
    ('Cautious', 'Wealthy', 'Rural') = 0.000001,
})

net.P['AntiTheft'] = pd.Series({
    # RA, SE, AT
    ('Psychopath', 'Prole', 'True') = 0.000001,
    ('Psychopath', 'Prole', 'False') = 0.999999,
    ('Adventurous', 'Prole', 'True') = 0.000001,
    ('Adventurous', 'Prole', 'False') = 0.999999,
    ('Normal', 'Prole', 'True') = 0.1,
    ('Normal', 'Prole', 'False') = 0.9,
    ('Cautious', 'Prole', 'True') = 0.95,
    ('Cautious', 'Prole', 'False') = 0.05,

    ('Psychopath', 'Middle', 'True') = 0.000001,
    ('Psychopath', 'Middle', 'False') = 0.999999,
    ('Adventurous', 'Middle', 'True') = 0.000001,
    ('Adventurous', 'Middle', 'False') = 0.999999,
    ('Normal', 'Middle', 'True') = 0.3,
    ('Normal', 'Middle', 'False') = 0.7,
    ('Cautious', 'Middle', 'True') = 0.999999,
    ('Cautious', 'Middle', 'False') = 0.000001,

    ('Psychopath', 'UpperMiddle', 'True') = 0.05,
    ('Psychopath', 'UpperMiddle', 'False') = 0.95,
    ('Adventurous', 'UpperMiddle', 'True') = 0.2,
    ('Adventurous', 'UpperMiddle', 'False') = 0.8,
    ('Normal', 'UpperMiddle', 'True') = 0.9,
    ('Normal', 'UpperMiddle', 'False') = 0.1,
    ('Cautious', 'UpperMiddle', 'True') = 0.999999,
    ('Cautious', 'UpperMiddle', 'False') = 0.000001,

    ('Psychopath', 'Wealthy', 'True') = 0.5,
    ('Psychopath', 'Wealthy', 'False') = 0.5,
    ('Adventurous', 'Wealthy', 'True') = 0.5,
    ('Adventurous', 'Wealthy', 'False') = 0.5,
    ('Normal', 'Wealthy', 'True') = 0.8,
    ('Normal', 'Wealthy', 'False') = 0.2,
    ('Cautious', 'Wealthy', 'True') = 0.999999,
    ('Cautious', 'Wealthy', 'False') = 0.000001,
})

net.P['DrivingSkill'] = pd.Series({
    # Age, #ST, #DS
    ('Adolescent','True', 'Substandard'): 0.5,
    ('Adolescent','True', 'Normal'): 0.45,
    ('Adolescent','True', 'Expert'): 0.05,
    ('Adult','True', 'Substandard'): 0.3,
    ('Adult','True', 'Normal'): 0.6,
    ('Adult','True', 'Expert'): 0.1,
    ('Senior','True', 'Substandard'): 0.1,
    ('Senior','True', 'Normal'): 0.6,
    ('Senior','True', 'Expert'): 0.3,

    ('Adolescent','False', 'Substandard'): 0.5,
    ('Adolescent','False', 'Normal'): 0.45,
    ('Adolescent','False', 'Expert'): 0.05,
    ('Adult','False', 'Substandard'): 0.3,
    ('Adult','False', 'Normal'): 0.6,
    ('Adult','False', 'Expert'): 0.1,
    ('Senior','False', 'Substandard'): 0.4,
    ('Senior','False', 'Normal'): 0.5,
    ('Senior','False', 'Expert'): 0.1,
})


net.P['DrivQuality'] = pd.Series({
    # Skill, #RA, #DQ
    ('Substandard', 'Psychopath', 'Poor'): 1.0,
    ('Substandard', 'Psychopath', 'Normal'): 0.0,
    ('Substandard', 'Psychopath', 'Excellent'): 0.0,
    ('Normal', 'Psychopath', 'Poor'): 0.5,
    ('Normal', 'Psychopath', 'Normal'): 0.2,
    ('Normal', 'Psychopath', 'Excellent'): 0.3,
    ('Expert', 'Psychopath', 'Poor'): 0.3,
    ('Expert', 'Psychopath', 'Normal'): 0.2,
    ('Expert', 'Psychopath', 'Excellent'): 0.5,

    ('Substandard', 'Adventurous', 'Poor'): 1.0,
    ('Substandard', 'Adventurous', 'Normal'): 0.0,
    ('Substandard', 'Adventurous', 'Excellent'): 0.0,
    ('Normal', 'Adventurous', 'Poor'): 0.3,
    ('Normal', 'Adventurous', 'Normal'): 0.4,
    ('Normal', 'Adventurous', 'Excellent'): 0.3,
    ('Expert', 'Adventurous', 'Poor'): 0.01,
    ('Expert', 'Adventurous', 'Normal'): 0.98,
    ('Expert', 'Adventurous', 'Excellent'): 0.01,

    ('Substandard', 'Normal', 'Poor'): 1.0,
    ('Substandard', 'Normal', 'Normal'): 0.0,
    ('Substandard', 'Normal', 'Excellent'): 0.0,
    ('Normal', 'Normal', 'Poor'): 0.0,
    ('Normal', 'Normal', 'Normal'): 1.0,
    ('Normal', 'Normal', 'Excellent'): 0.0,
    ('Expert', 'Normal', 'Poor'): 0.0,
    ('Expert', 'Normal', 'Normal'): 0.0,
    ('Expert', 'Normal', 'Excellent'): 1.0,

    ('Substandard', 'Cautious', 'Poor'): 1.0,
    ('Substandard', 'Cautious', 'Normal'): 0.0,
    ('Substandard', 'Cautious', 'Excellent'): 0.0,
    ('Normal', 'Cautious', 'Poor'): 0.0,
    ('Normal', 'Cautious', 'Normal'): 0.8,
    ('Normal', 'Cautious', 'Excellent'): 0.2,
    ('Expert', 'Cautious', 'Poor'): 0.9,
    ('Expert', 'Cautious', 'Normal'): 0.0,
    ('Expert', 'Cautious', 'Excellent'): 1.0,
})

net.P['MakeModel'] = pd.Series({
    #SE, RA, MM
    ('Prole', 'Psychopath', 'Sportscar') = 0.1,
    ('Prole', 'Psychopath', 'Economy') = 0.7,
    ('Prole', 'Psychopath', 'FamilySedan') = 0.2,
    ('Prole', 'Psychopath', 'Luxury') = 0.0,
    ('Prole', 'Psychopath', 'SuperLuxury') = 0.0,

    ('Middle', 'Psychopath', 'Sportscar') = 0.15,
    ('Middle', 'Psychopath', 'Economy') = 0.20,
    ('Middle', 'Psychopath', 'FamilySedan') = 0.65,
    ('Middle', 'Psychopath', 'Luxury') = 0.0,
    ('Middle', 'Psychopath', 'SuperLuxury') = 0.0,

    ('UpperMiddle', 'Psychopath', 'Sportscar') = 0.20,
    ('UpperMiddle', 'Psychopath', 'Economy') = 0.05,
    ('UpperMiddle', 'Psychopath', 'FamilySedan') = 0.30,
    ('UpperMiddle', 'Psychopath', 'Luxury') = 0.45,
    ('UpperMiddle', 'Psychopath', 'SuperLuxury') = 0.0,

    ('Wealthy', 'Psychopath', 'Sportscar') = 0.30,
    ('Wealthy', 'Psychopath', 'Economy') = 0.01,
    ('Wealthy', 'Psychopath', 'FamilySedan') = 0.09,
    ('Wealthy', 'Psychopath', 'Luxury') = 0.40,
    ('Wealthy', 'Psychopath', 'SuperLuxury') = 0.20,


    ('Prole', 'Adventurous', 'Sportscar') = 0.1,
    ('Prole', 'Adventurous', 'Economy') = 0.7,
    ('Prole', 'Adventurous', 'FamilySedan') = 0.2,
    ('Prole', 'Adventurous', 'Luxury') = 0.0,
    ('Prole', 'Adventurous', 'SuperLuxury') = 0.0,

    ('Middle', 'Adventurous', 'Sportscar') = 0.15,
    ('Middle', 'Adventurous', 'Economy') = 0.20,
    ('Middle', 'Adventurous', 'FamilySedan') = 0.65,
    ('Middle', 'Adventurous', 'Luxury') = 0.0,
    ('Middle', 'Adventurous', 'SuperLuxury') = 0.0,

    ('UpperMiddle', 'Adventurous', 'Sportscar') = 0.20,
    ('UpperMiddle', 'Adventurous', 'Economy') = 0.05,
    ('UpperMiddle', 'Adventurous', 'FamilySedan') = 0.30,
    ('UpperMiddle', 'Adventurous', 'Luxury') = 0.45,
    ('UpperMiddle', 'Adventurous', 'SuperLuxury') = 0.0,

    ('Wealthy', 'Adventurous', 'Sportscar') = 0.30,
    ('Wealthy', 'Adventurous', 'Economy') = 0.01,
    ('Wealthy', 'Adventurous', 'FamilySedan') = 0.09,
    ('Wealthy', 'Adventurous', 'Luxury') = 0.40,
    ('Wealthy', 'Adventurous', 'SuperLuxury') = 0.20,

    ('Prole', 'Normal', 'Sportscar') = 0.1,
    ('Prole', 'Normal', 'Economy') = 0.7,
    ('Prole', 'Normal', 'FamilySedan') = 0.2,
    ('Prole', 'Normal', 'Luxury') = 0.0,
    ('Prole', 'Normal', 'SuperLuxury') = 0.0,

    ('Middle', 'Normal', 'Sportscar') = 0.15,
    ('Middle', 'Normal', 'Economy') = 0.20,
    ('Middle', 'Normal', 'FamilySedan') = 0.65,
    ('Middle', 'Normal', 'Luxury') = 0.0,
    ('Middle', 'Normal', 'SuperLuxury') = 0.0,

    ('UpperMiddle', 'Normal', 'Sportscar') = 0.20,
    ('UpperMiddle', 'Normal', 'Economy') = 0.05,
    ('UpperMiddle', 'Normal', 'FamilySedan') = 0.30,
    ('UpperMiddle', 'Normal', 'Luxury') = 0.45,
    ('UpperMiddle', 'Normal', 'SuperLuxury') = 0.0,

    ('Wealthy', 'Normal', 'Sportscar') = 0.30,
    ('Wealthy', 'Normal', 'Economy') = 0.01,
    ('Wealthy', 'Normal', 'FamilySedan') = 0.09,
    ('Wealthy', 'Normal', 'Luxury') = 0.40,
    ('Wealthy', 'Normal', 'SuperLuxury') = 0.20,
    
    ('Prole', 'Cautious', 'Sportscar') = 0.1,
    ('Prole', 'Cautious', 'Economy') = 0.7,
    ('Prole', 'Cautious', 'FamilySedan') = 0.2,
    ('Prole', 'Cautious', 'Luxury') = 0.0,
    ('Prole', 'Cautious', 'SuperLuxury') = 0.0,

    ('Middle', 'Cautious', 'Sportscar') = 0.15,
    ('Middle', 'Cautious', 'Economy') = 0.20,
    ('Middle', 'Cautious', 'FamilySedan') = 0.65,
    ('Middle', 'Cautious', 'Luxury') = 0.0,
    ('Middle', 'Cautious', 'SuperLuxury') = 0.0,

    ('UpperMiddle', 'Cautious', 'Sportscar') = 0.20,
    ('UpperMiddle', 'Cautious', 'Economy') = 0.05,
    ('UpperMiddle', 'Cautious', 'FamilySedan') = 0.30,
    ('UpperMiddle', 'Cautious', 'Luxury') = 0.45,
    ('UpperMiddle', 'Cautious', 'SuperLuxury') = 0.0,

    ('Wealthy', 'Cautious', 'Sportscar') = 0.30,
    ('Wealthy', 'Cautious', 'Economy') = 0.01,
    ('Wealthy', 'Cautious', 'FamilySedan') = 0.09,
    ('Wealthy', 'Cautious', 'Luxury') = 0.40,
    ('Wealthy', 'Cautious', 'SuperLuxury') = 0.20,
})

net.P['VehicleYear'] = pd.Series({
    #SE, RA, VY
    ('Prole', 'Psychopath', 'Current') = 0.15,
    ('Prole', 'Psychopath', 'Older') = 0.85,
    ('Middle', 'Psychopath', 'Current') = 0.3,
    ('Middle', 'Psychopath', 'Older') = 0.7,
    ('UpperMiddle', 'Psychopath', 'Current') = 0.8,
    ('UpperMiddle', 'Psychopath', 'Older') = 0.2,
    ('Wealthy', 'Psychopath', 'Current') = 0.9,
    ('Wealthy', 'Psychopath', 'Older') = 0.1,

    ('Prole', 'Adventurous', 'Current') = 0.15,
    ('Prole', 'Adventurous', 'Older') = 0.85,
    ('Middle', 'Adventurous', 'Current') = 0.3,
    ('Middle', 'Adventurous', 'Older') = 0.7,
    ('UpperMiddle', 'Adventurous', 'Current') = 0.8,
    ('UpperMiddle', 'Adventurous', 'Older') = 0.2,
    ('Wealthy', 'Adventurous', 'Current') = 0.9,
    ('Wealthy', 'Adventurous', 'Older') = 0.1,

    ('Prole', 'Normal', 'Current') = 0.15,
    ('Prole', 'Normal', 'Older') = 0.85,
    ('Middle', 'Normal', 'Current') = 0.3,
    ('Middle', 'Normal', 'Older') = 0.7,
    ('UpperMiddle', 'Normal', 'Current') = 0.8,
    ('UpperMiddle', 'Normal', 'Older') = 0.2,
    ('Wealthy', 'Normal', 'Current') = 0.9,
    ('Wealthy', 'Normal', 'Older') = 0.1,

    ('Prole', 'Cautious', 'Current') = 0.15,
    ('Prole', 'Cautious', 'Older') = 0.85,
    ('Middle', 'Cautious', 'Current') = 0.3,
    ('Middle', 'Cautious', 'Older') = 0.7,
    ('UpperMiddle', 'Cautious', 'Current') = 0.8,
    ('UpperMiddle', 'Cautious', 'Older') = 0.2,
    ('Wealthy', 'Cautious', 'Current') = 0.9,
    ('Wealthy', 'Cautious', 'Older') = 0.1,
})

net.P['Mileage'] = pd.Series({'FiveThou': 0.1, 'TwentyThou': 0.4, 'FiftyThou': 0.4, 'Domino': 0.1})

net.P['Airbag'] = pd.Series({
    #MM, VY, AB
    ('Sportscar', 'Current', 'True') = 1.0,
    ('Sportscar', 'Current', 'False') = 0.0,
    ('Economy', 'Current', 'True') = 1.0,
    ('Economy', 'Current', 'False') = 0.0,
    ('FamilySedan', 'Current', 'True') = 1.0,
    ('FamilySedan', 'Current', 'False') = 0.0,
    ('Luxury', 'Current', 'True') = 1.0,
    ('Luxury', 'Current', 'False') = 0.0,
    ('SuperLuxury', 'Current','True') = 1.0,
    ('SuperLuxury', 'Current', 'False') = 0.0,

    ('Sportscar', 'Older', 'True') = 0.1,
    ('Sportscar', 'Older', 'False') = 0.9,
    ('Economy', 'Older', 'True') = 0.05,
    ('Economy', 'Older', 'False') = 0.95,
    ('FamilySedan', 'Older', 'True') = 0.2,
    ('FamilySedan', 'Older', 'False') = 0.8,
    ('Luxury', 'Older', 'True') = 0.6,
    ('Luxury', 'Older', 'False') = 0.4,
    ('SuperLuxury', 'Older','True') = 0.1,
    ('SuperLuxury', 'Older', 'False') = 0.9,

})


net.P['Antilock'] = pd.Series({
    #MM, VY, Al
    ('Sportscar', 'Current', 'True') = 0.9,
    ('Sportscar', 'Current', 'False') = 0.1,
    ('Economy', 'Current', 'True') = 0.001,
    ('Economy', 'Current', 'False') = 0.999,
    ('FamilySedan', 'Current', 'True') = 0.4,
    ('FamilySedan', 'Current', 'False') = 0.6,
    ('Luxury', 'Current', 'True') = 0.99,
    ('Luxury', 'Current', 'False') = 0.01,
    ('SuperLuxury', 'Current','True') = 0.99,
    ('SuperLuxury', 'Current', 'False') = 0.01,

    ('Sportscar', 'Older', 'True') = 0.1,
    ('Sportscar', 'Older', 'False') = 0.9,
    ('Economy', 'Older', 'True') = 0.0,
    ('Economy', 'Older', 'False') = 1.0,
    ('FamilySedan', 'Older', 'True') = 0.0,
    ('FamilySedan', 'Older', 'False') = 1.0,
    ('Luxury', 'Older', 'True') = 0.3,
    ('Luxury', 'Older', 'False') = 0.7,
    ('SuperLuxury', 'Older','True') = 0.15,
    ('SuperLuxury', 'Older', 'False') = 0.85,

})


net.P['RuggedAuto'] = pd.Series({
    #MM, VY, RGA
    ('Sportscar', 'Current', 'EggShell') = 0.95,
    ('Sportscar', 'Current', 'Football') = 0.04,
    ('Sportscar', 'Current', 'Tank') = 0.01,
    ('Economy', 'Current', 'EggShell') = 0.5,
    ('Economy', 'Current', 'Football') = 0.5,
    ('Economy', 'Current', 'Tank') = 0.0,
    ('FamilySedan', 'Current', 'EggShell') = 0.2,
    ('FamilySedan', 'Current', 'Football') = 0.6,
    ('FamilySedan', 'Current', 'Tank') = 0.2,
    ('Luxury', 'Current', 'EggShell') = 0.1,
    ('Luxury', 'Current', 'Football') = 0.6,
    ('Luxury', 'Current', 'Tank') = 0.3,
    ('SuperLuxury', 'Current','EggShell') = 0.05,
    ('SuperLuxury', 'Current', 'Football') = 0.55,
    ('SuperLuxury', 'Current', 'Tank') = 0.4,

    ('Sportscar', 'Older', 'EggShell') = 0.95,
    ('Sportscar', 'Older', 'Football') = 0.04,
    ('Sportscar', 'Older', 'Tank') = 0.01,
    ('Economy', 'Older', 'EggShell') = 0.9,
    ('Economy', 'Older', 'Football') = 0.1,
    ('Economy', 'Older', 'Tank') = 0.0,
    ('FamilySedan', 'Older', 'EggShell') = 0.05,
    ('FamilySedan', 'Older', 'Football') = 0.55,
    ('FamilySedan', 'Older', 'Tank') = 0.4,
    ('Luxury', 'Older', 'EggShell') = 0.1,
    ('Luxury', 'Older', 'Football') = 0.6,
    ('Luxury', 'Older', 'Tank') = 0.3,
    ('SuperLuxury', 'Older','EggShell') = 0.05,
    ('SuperLuxury', 'Older', 'Football') = 0.55,
    ('SuperLuxury', 'Older', 'Tank') = 0.4,

})

net.P['CarValue'] = pd.Series({
    # MM, #VY, #M ('FiveThou': 0.1, 'TwentyThou': 0.4, 'FiftyThou': 0.4, 'Domino': 0.1}))
    ('SportsCar', 'Current', 'FiveThou', 'FiveThou') = 0.0,
    ('SportsCar', 'Current', 'FiveThou', 'TenThou') = 0.1,
    ('SportsCar', 'Current', 'FiveThou', 'TwentyThou') = 0.8,
    ('SportsCar', 'Current', 'FiveThou', 'FiftyThou') = 0.09,
    ('SportsCar', 'Current', 'FiveThou', 'Million') = 0.01,

    ('Economy', 'Current', 'FiveThou', 'FiveThou') = 0.1,
    ('Economy', 'Current', 'FiveThou', 'TenThou') = 0.8,
    ('Economy', 'Current', 'FiveThou', 'TwentyThou') = 0.1,
    ('Economy', 'Current', 'FiveThou', 'FiftyThou') = 0.0,
    ('Economy', 'Current', 'FiveThou', 'Million') = 0.0,

    ('FamilySedan', 'Current', 'FiveThou', 'FiveThou') = 0.0,
    ('FamilySedan', 'Current', 'FiveThou', 'TenThou') = 0.1,
    ('FamilySedan', 'Current', 'FiveThou', 'TwentyThou') = 0.9,
    ('FamilySedan', 'Current', 'FiveThou', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Current', 'FiveThou', 'Million') = 0.0,

    ('Luxury', 'Current', 'FiveThou', 'FiveThou') = 0.0,
    ('Luxury', 'Current', 'FiveThou', 'TenThou') = 0.0,
    ('Luxury', 'Current', 'FiveThou', 'TwentyThou') = 0.0,
    ('Luxury', 'Current', 'FiveThou', 'FiftyThou') = 1.0,
    ('Luxury', 'Current', 'FiveThou', 'Million') = 0.0,

    ('SuperLuxury', 'Current', 'FiveThou', 'FiveThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiveThou', 'TenThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiveThou', 'TwentyThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiveThou', 'FiftyThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiveThou', 'Million') = 1.0,


    ('SportsCar', 'Older', 'FiveThou', 'FiveThou') = 0.03,
    ('SportsCar', 'Older', 'FiveThou', 'TenThou') = 0.3,
    ('SportsCar', 'Older', 'FiveThou', 'TwentyThou') = 0.6,
    ('SportsCar', 'Older', 'FiveThou', 'FiftyThou') = 0.06,
    ('SportsCar', 'Older', 'FiveThou', 'Million') = 0.01,

    ('Economy', 'Older', 'FiveThou', 'FiveThou') = 0.25,
    ('Economy', 'Older', 'FiveThou', 'TenThou') = 0.7,
    ('Economy', 'Older', 'FiveThou', 'TwentyThou') = 0.05,
    ('Economy', 'Older', 'FiveThou', 'FiftyThou') = 0.0,
    ('Economy', 'Older', 'FiveThou', 'Million') = 0.0,

    ('FamilySedan', 'Older', 'FiveThou', 'FiveThou') = 0.2,
    ('FamilySedan', 'Older', 'FiveThou', 'TenThou') = 0.3,
    ('FamilySedan', 'Older', 'FiveThou', 'TwentyThou') = 0.5,
    ('FamilySedan', 'Older', 'FiveThou', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Older', 'FiveThou', 'Million') = 0.0,

    ('Luxury', 'Older', 'FiveThou', 'FiveThou') = 0.01,
    ('Luxury', 'Older', 'FiveThou', 'TenThou') = 0.09,
    ('Luxury', 'Older', 'FiveThou', 'TwentyThou') = 0.20,
    ('Luxury', 'Older', 'FiveThou', 'FiftyThou') = 0.70,
    ('Luxury', 'Older', 'FiveThou', 'Million') = 0.0,

    ('SuperLuxury', 'Older', 'FiveThou', 'FiveThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiveThou', 'TenThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiveThou', 'TwentyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiveThou', 'FiftyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiveThou', 'Million') = 0.999996,


    ('SportsCar', 'Current', 'TwentyThou', 'FiveThou') = 0.0,
    ('SportsCar', 'Current', 'TwentyThou', 'TenThou') = 0.1,
    ('SportsCar', 'Current', 'TwentyThou', 'TwentyThou') = 0.8,
    ('SportsCar', 'Current', 'TwentyThou', 'FiftyThou') = 0.09,
    ('SportsCar', 'Current', 'TwentyThou', 'Million') = 0.01,

    ('Economy', 'Current', 'TwentyThou', 'FiveThou') = 0.1,
    ('Economy', 'Current', 'TwentyThou', 'TenThou') = 0.8,
    ('Economy', 'Current', 'TwentyThou', 'TwentyThou') = 0.1,
    ('Economy', 'Current', 'TwentyThou', 'FiftyThou') = 0.0,
    ('Economy', 'Current', 'TwentyThou', 'Million') = 0.0,

    ('FamilySedan', 'Current', 'TwentyThou', 'FiveThou') = 0.0,
    ('FamilySedan', 'Current', 'TwentyThou', 'TenThou') = 0.1,
    ('FamilySedan', 'Current', 'TwentyThou', 'TwentyThou') = 0.9,
    ('FamilySedan', 'Current', 'TwentyThou', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Current', 'TwentyThou', 'Million') = 0.0,

    ('Luxury', 'Current', 'TwentyThou', 'FiveThou') = 0.0,
    ('Luxury', 'Current', 'TwentyThou', 'TenThou') = 0.0,
    ('Luxury', 'Current', 'TwentyThou', 'TwentyThou') = 0.0,
    ('Luxury', 'Current', 'TwentyThou', 'FiftyThou') = 1.0,
    ('Luxury', 'Current', 'TwentyThou', 'Million') = 0.0,

    ('SuperLuxury', 'Current', 'TwentyThou', 'FiveThou') = 0.0,
    ('SuperLuxury', 'Current', 'TwentyThou', 'TenThou') = 0.0,
    ('SuperLuxury', 'Current', 'TwentyThou', 'TwentyThou') = 0.0,
    ('SuperLuxury', 'Current', 'TwentyThou', 'FiftyThou') = 0.0,
    ('SuperLuxury', 'Current', 'TwentyThou', 'Million') = 1.0,


    ('SportsCar', 'Older', 'TwentyThou', 'FiveThou') = 0.16,
    ('SportsCar', 'Older', 'TwentyThou', 'TenThou') = 0.5,
    ('SportsCar', 'Older', 'TwentyThou', 'TwentyThou') = 0.3,
    ('SportsCar', 'Older', 'TwentyThou', 'FiftyThou') = 0.03,
    ('SportsCar', 'Older', 'TwentyThou', 'Million') = 0.01,

    ('Economy', 'Older', 'TwentyThou', 'FiveThou') = 0.7,
    ('Economy', 'Older', 'TwentyThou', 'TenThou') = 0.2999,
    ('Economy', 'Older', 'TwentyThou', 'TwentyThou') = 0.0001,
    ('Economy', 'Older', 'TwentyThou', 'FiftyThou') = 0.0,
    ('Economy', 'Older', 'TwentyThou', 'Million') = 0.0,

    ('FamilySedan', 'Older', 'TwentyThou', 'FiveThou') = 0.5,
    ('FamilySedan', 'Older', 'TwentyThou', 'TenThou') = 0.3
    ('FamilySedan', 'Older', 'TwentyThou', 'TwentyThou') = 0.2,
    ('FamilySedan', 'Older', 'TwentyThou', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Older', 'TwentyThou', 'Million') = 0.0,

    ('Luxury', 'Older', 'TwentyThou', 'FiveThou') = 0.05,
    ('Luxury', 'Older', 'TwentyThou', 'TenThou') = 0.15,
    ('Luxury', 'Older', 'TwentyThou', 'TwentyThou') = 0.30,
    ('Luxury', 'Older', 'TwentyThou', 'FiftyThou') = 0.50,
    ('Luxury', 'Older', 'TwentyThou', 'Million') = 0.0,

    ('SuperLuxury', 'Older', 'TwentyThou', 'FiveThou') = 0.000001,
    ('SuperLuxury', 'Older', 'TwentyThou', 'TenThou') = 0.000001,
    ('SuperLuxury', 'Older', 'TwentyThou', 'TwentyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'TwentyThou', 'FiftyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'TwentyThou', 'Million') = 0.999996,

    ('SportsCar', 'Current', 'FiftyThou', 'FiveThou') = 0.0,
    ('SportsCar', 'Current', 'FiftyThou', 'TenThou') = 0.1,
    ('SportsCar', 'Current', 'FiftyThou', 'TwentyThou') = 0.8,
    ('SportsCar', 'Current', 'FiftyThou', 'FiftyThou') = 0.09,
    ('SportsCar', 'Current', 'FiftyThou', 'Million') = 0.01,

    ('Economy', 'Current', 'FiftyThou', 'FiveThou') = 0.1,
    ('Economy', 'Current', 'FiftyThou', 'TenThou') = 0.8,
    ('Economy', 'Current', 'FiftyThou', 'TwentyThou') = 0.1,
    ('Economy', 'Current', 'FiftyThou', 'FiftyThou') = 0.0,
    ('Economy', 'Current', 'FiftyThou', 'Million') = 0.0,

    ('FamilySedan', 'Current', 'FiftyThou', 'FiveThou') = 0.0,
    ('FamilySedan', 'Current', 'FiftyThou', 'TenThou') = 0.1,
    ('FamilySedan', 'Current', 'FiftyThou', 'TwentyThou') = 0.9,
    ('FamilySedan', 'Current', 'FiftyThou', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Current', 'FiftyThou', 'Million') = 0.0,

    ('Luxury', 'Current', 'FiftyThou', 'FiveThou') = 0.0,
    ('Luxury', 'Current', 'FiftyThou', 'TenThou') = 0.0,
    ('Luxury', 'Current', 'FiftyThou', 'TwentyThou') = 0.0,
    ('Luxury', 'Current', 'FiftyThou', 'FiftyThou') = 1.0,
    ('Luxury', 'Current', 'FiftyThou', 'Million') = 0.0,

    ('SuperLuxury', 'Current', 'FiftyThou', 'FiveThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiftyThou', 'TenThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiftyThou', 'TwentyThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiftyThou', 'FiftyThou') = 0.0,
    ('SuperLuxury', 'Current', 'FiftyThou', 'Million') = 1.0,


    ('SportsCar', 'Older', 'FiftyThou', 'FiveThou') = 0.4,
    ('SportsCar', 'Older', 'FiftyThou', 'TenThou') = 0.47,
    ('SportsCar', 'Older', 'FiftyThou', 'TwentyThou') = 0.10,
    ('SportsCar', 'Older', 'FiftyThou', 'FiftyThou') = 0.02,
    ('SportsCar', 'Older', 'FiftyThou', 'Million') = 0.01,

    ('Economy', 'Older', 'FiftyThou', 'FiveThou') = 0.990000,
    ('Economy', 'Older', 'FiftyThou', 'TenThou') = 0.009999,
    ('Economy', 'Older', 'FiftyThou', 'TwentyThou') = 0.000001,
    ('Economy', 'Older', 'FiftyThou', 'FiftyThou') = 0.0,
    ('Economy', 'Older', 'FiftyThou', 'Million') = 0.0,

    ('FamilySedan', 'Older', 'FiftyThou', 'FiveThou') = 0.7,
    ('FamilySedan', 'Older', 'FiftyThou', 'TenThou') = 0.2,
    ('FamilySedan', 'Older', 'FiftyThou', 'TwentyThou') = 0.1,
    ('FamilySedan', 'Older', 'FiftyThou', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Older', 'FiftyThou', 'Million') = 0.0,

    ('Luxury', 'Older', 'FiftyThou', 'FiveThou') = 0.1,
    ('Luxury', 'Older', 'FiftyThou', 'TenThou') = 0.3,
    ('Luxury', 'Older', 'FiftyThou', 'TwentyThou') = 0.3,
    ('Luxury', 'Older', 'FiftyThou', 'FiftyThou') = 0.3,
    ('Luxury', 'Older', 'FiftyThou', 'Million') = 0.0,

    ('SuperLuxury', 'Older', 'FiftyThou', 'FiveThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiftyThou', 'TenThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiftyThou', 'TwentyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiftyThou', 'FiftyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'FiftyThou', 'Million') = 0.999996,


    ('SportsCar', 'Current', 'Domino', 'FiveThou') = 0.0,
    ('SportsCar', 'Current', 'Domino', 'TenThou') = 0.1,
    ('SportsCar', 'Current', 'Domino', 'TwentyThou') = 0.8,
    ('SportsCar', 'Current', 'Domino', 'FiftyThou') = 0.09,
    ('SportsCar', 'Current', 'Domino', 'Million') = 0.01,

    ('Economy', 'Current', 'Domino', 'FiveThou') = 0.1,
    ('Economy', 'Current', 'Domino', 'TenThou') = 0.8,
    ('Economy', 'Current', 'Domino', 'TwentyThou') = 0.1,
    ('Economy', 'Current', 'Domino', 'FiftyThou') = 0.0,
    ('Economy', 'Current', 'Domino', 'Million') = 0.0,

    ('FamilySedan', 'Current', 'Domino', 'FiveThou') = 0.0,
    ('FamilySedan', 'Current', 'Domino', 'TenThou') = 0.1,
    ('FamilySedan', 'Current', 'Domino', 'TwentyThou') = 0.9,
    ('FamilySedan', 'Current', 'Domino', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Current', 'Domino', 'Million') = 0.0,

    ('Luxury', 'Current', 'Domino', 'FiveThou') = 0.0,
    ('Luxury', 'Current', 'Domino', 'TenThou') = 0.0,
    ('Luxury', 'Current', 'Domino', 'TwentyThou') = 0.0,
    ('Luxury', 'Current', 'Domino', 'FiftyThou') = 1.0,
    ('Luxury', 'Current', 'Domino', 'Million') = 0.0,

    ('SuperLuxury', 'Current', 'Domino', 'FiveThou') = 0.0,
    ('SuperLuxury', 'Current', 'Domino', 'TenThou') = 0.0,
    ('SuperLuxury', 'Current', 'Domino', 'TwentyThou') = 0.0,
    ('SuperLuxury', 'Current', 'Domino', 'FiftyThou') = 0.0,
    ('SuperLuxury', 'Current', 'Domino', 'Million') = 1.0,

    ('SportsCar', 'Older', 'Domino', 'FiveThou') = 0.9,
    ('SportsCar', 'Older', 'Domino', 'TenThou') = 0.06,
    ('SportsCar', 'Older', 'Domino', 'TwentyThou') = 0.02,
    ('SportsCar', 'Older', 'Domino', 'FiftyThou') = 0.01,
    ('SportsCar', 'Older', 'Domino', 'Million') = 0.01,

    ('Economy', 'Older', 'Domino', 'FiveThou') = 0.999998,
    ('Economy', 'Older', 'Domino', 'TenThou') = 0.000001,
    ('Economy', 'Older', 'Domino', 'TwentyThou') = 0.000001,
    ('Economy', 'Older', 'Domino', 'FiftyThou') = 0.0,
    ('Economy', 'Older', 'Domino', 'Million') = 0.0,

    ('FamilySedan', 'Older', 'Domino', 'FiveThou') = 0.990000,
    ('FamilySedan', 'Older', 'Domino', 'TenThou') = 0.009999,
    ('FamilySedan', 'Older', 'Domino', 'TwentyThou') = 0.000001,
    ('FamilySedan', 'Older', 'Domino', 'FiftyThou') = 0.0,
    ('FamilySedan', 'Older', 'Domino', 'Million') = 0.0,

    ('Luxury', 'Older', 'Domino', 'FiveThou') = 0.2,
    ('Luxury', 'Older', 'Domino', 'TenThou') = 0.2,
    ('Luxury', 'Older', 'Domino', 'TwentyThou') = 0.3,
    ('Luxury', 'Older', 'Domino', 'FiftyThou') = 0.3,
    ('Luxury', 'Older', 'Domino', 'Million') = 0.0,

    ('SuperLuxury', 'Older', 'Domino', 'FiveThou') = 0.000001,
    ('SuperLuxury', 'Older', 'Domino', 'TenThou') = 0.000001,
    ('SuperLuxury', 'Older', 'Domino', 'TwentyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'Domino', 'FiftyThou') = 0.000001,
    ('SuperLuxury', 'Older', 'Domino', 'Million') = 0.999996,
})


Insurance_Network = net