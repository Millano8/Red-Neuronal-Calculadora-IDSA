# El siguiente código crea una red neuronal que aprende a sumar, restar, dividir y multiplicar


import numpy as np
import pandas as pd
import random


# Paso 1
def get_random_ops(rows=100):
    data = []
    for i in range(0,rows):
        a = random.randint(1,100)
        b= random.randint(1,100)
        suma,resta,multiplicacion,division = random.choice([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ])

        if suma == 1:
            y = a+b
        elif resta == 1:
            y = a-b
        elif multiplicacion == 1:
            y = a*b
        elif division == 1:
            y = a/b 
        
        data.append({
            "a":a,
            "b":b,
            "suma":suma,
            "resta":resta,
            "multiplicacion":multiplicacion,
            "division":division,
            "y":round(y,2)
        })
    return data

#print(get_random_ops())
        
# Paso 2

data = pd.DataFrame(get_random_ops(rows=100000))
print(data[["a","b","suma","resta","multiplicacion","division","y"]].head())
print(data.hist())
print(data.shape)

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 

X_train,X_test,Y_train,Y_test = train_test_split(data[["a","b","suma","resta","multiplicacion","division"]],
                                                 data["y"],test_size=0.30,random_state=42)

model = MLPRegressor(max_iter=800,hidden_layer_sizes=(100,100,100),learning_rate_init=0.0001)

model.fit(X_train,Y_train) 

print(X_test.iloc[50])
print(Y_test.iloc[50])

print(model.predict([X_test.iloc[50]]))
predict = model.predict(X_test)

print("Predict: %s" % list(predict[:5]))

data_check = pd.DataFrame(predict,columns=["predict"])
data_check["y"] = list(Y_test)
data_check.set_index(["y"],drop=False,inplace=True)
data_check.sort_values(by=["y"],inplace=True)

data_check.plot()