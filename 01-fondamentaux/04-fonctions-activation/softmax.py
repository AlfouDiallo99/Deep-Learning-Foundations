import numpy as np
#softmax donne une sorte de probabiltée tres utile en classification
def Softmax(x):
  exp_x=np.exp(x-np.max(x))
  return exp_x/np.sum(exp_x)