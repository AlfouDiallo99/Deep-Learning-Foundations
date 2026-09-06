# implementation de la fonction de perte ,l'erreur quadratique moyenne 
# Fonction de perte :
#
#             m
# L = 1/(2m) Σ (ŷᵢ − yᵢ)²
#  
import numpy as np
def MSE(y_true,y_predict):
  return np.sum((y_predict-y_true)**2)/(2*y_true.shape[1])
#derivee de la fonction
def MSE_derivee(y_true,y_predict):
  return (1/(y_true.shape[1]))*(y_predict-y_true)

