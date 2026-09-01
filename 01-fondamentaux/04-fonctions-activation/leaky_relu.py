import numpy as np
#leaky relu est une variante de Relu qui permet d'evite les neurones mortes lors du backpropagation avec le gradient Descent
def Leaky_relu(x,alpha=0.01):
  return np.where(x>0,x,alpha*x)
