import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('day.csv')

#batch gradient descent

def gradient_descent_regression(X,Y,m_init,b_init,learning_rate,max_iters):
    m=[m_init]
    b=[b_init]
    error=[0]
    N=len(X)
    descending = True
    iters = 0
    while descending:
        e=0   
        p_m=0
        p_b=0
        b_i=b[iters]
        m_i=m[iters]
        for i in range(0,N):
            yhat=m_i*X[i]+b_i
            e+=(Y[i]-yhat)**2
            p_m+=(2/N)*(-X[i]*(Y[i]-yhat))
            p_b+=(-2/N)*(Y[i]-yhat)
        error.append(e/N)
        iters+=1
        b.append(b[iters-1]-(p_b*learning_rate))
        m.append(m[iters-1]-(p_m*learning_rate))
        if error[iters]>10000 or iters>= max_iters:
            print("The process has failed to converge")
            break
        if error[iters]<0.001:
            descending=False 
    return m,b,error,iters

x=np.sort(data['temp'])
y=np.sort(data['hum'])

m,b,error,iters=gradient_descent_regression(x,y,2,2,0.0001,25000)

y_hat=m[iters]*x+b[iters]

plt.plot(x,y)
plt.plot(x,y_hat)
plt.show()
