import numpy as np
#Elu est une variante de Relu , qui elle aussi permet d'eviter les neurones mortes lors du Backward

def Elu(x,alpha=.01):
 return np.where(x>0,x,alpha*(np.exp(x)-1))