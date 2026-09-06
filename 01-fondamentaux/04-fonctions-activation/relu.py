import numpy as np
def Relu(x):
  return np.maximum(0,x)



#derivee de relu
def Relu_derivee(x):
  return (x>0).astype(float)
