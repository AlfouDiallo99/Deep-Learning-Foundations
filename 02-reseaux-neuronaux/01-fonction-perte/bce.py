#implementation de la fonction de parte de 
#Entropie croisse binaire
import numpy as np

def Binary_cross_entropy(y_true,y_predict):
  #definition d'un epsilon pour eviter log(0) ou log(1)
  epsilon=1e-15
  m=y_true.shape[1]
  y_predict=np.clip(y_predict,epsilon,1-epsilon)
  return -np.sum(y_true*np.log(y_predict)+(1-y_true)*np.log(1-y_predict))/m
#la derivee 
def derivec_bce(y_true,y_predict):
   epsilon=1e-15
   y_predict=np.clip(y_predict,epsilon,1-epsilon)
   m=y_true.shape[1]
   return -1*(y_true*(1/y_predict)-(1-y_true)*(1/(1-y_predict)))/m