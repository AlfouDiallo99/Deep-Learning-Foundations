#dans cette parte ,on prevois de faire la retropropagation du Deeplearning avec numpy manuellement juste que 3 ou 4 couche et la couche de sortie ,eventuellement la prediction du reseaux
import numpy as np
import sys 
#import des fonction d'activation
sys.path.append("../../01-fondamentaux/04-fonctions-activation")
#import de relu
from relu import Relu,Relu_derivee
#import des fonctions de perte 
sys.path.append("../01-fonction-perte")
from mse import MSE,MSE_derivee
#creation de a premier couche 
X=np.array([[1,2],[2,1]])
W1=np.array([[1,-1],[2,1]])
b1=np.array([[0],[1]])
Z1=np.dot(W1,X)+b1
#resultat de la sortie
print(Z1)#resultats de la sortie sans la fonction d'activation
#application de la fonction relu
A1=Relu(Z1)  #application de Relu
print(A1) #resultat de relu sur la sortie Z
#Deuxieme couche
W2=np.array([[1,2],[-1,1]])
b2=np.array([[1],[0]])
Z2=np.dot(W2,A1)+b2
print(f"resultat de la sortie du layers number two{Z2}")
#application de la fonction d'activation 
A2=Relu(Z2)
print(f"Relu(Z2)={A2}")
#layer number three 
W3=np.array([[2,-1]])
b3=np.array([1])
Z3=np.dot(W3,A2)+b3
print(f"sortie Z3={Z3}")
#application de relu sur le sortie de la couche 3
A3=Relu(Z3)
print(f"Relu(Z3)={A3}")
#la couche de sortie , la prediction Y_predict
W4=np.array([[1]])
b4=np.array([[0]])
Z4=np.dot(W4,A3)+b4
print(f"Z4={Z4}")
A4=Z4 # prediction du reseaux ici pas de fonction d'action car c'est un regression linéaire
print(f"Y_predcit={A4}")
#la cible 
Y_true=np.array([[10,12]])
print(f"cost={MSE(Y_true,A4)}")
#la backpropagation pour minimisé la perte
#calculs des derivees partielles 
dA4=MSE_derivee(Y_true,A4)
print(f"Derivee L par rapport a Y_predict:{dA4}")
#derivee de L par rapport a Z4=derivee de L par rapport a A4
dZ4=MSE_derivee(Y_true,A4)
print(f"dZ4={dZ4}")
#la derivee du loss par rapport a W4
dW4=np.dot(dZ4,A3.T)
print(f"la derive du cost par rapport a W4={dW4}")
#derivee de b4 
db4 = np.sum(dZ4, axis=1, keepdims=True)
print(f"la derivee du cost par rapport a b4:{db4}")
#derivee du loss par rapport a A3
dA3=np.dot(W4.T,dZ4)
print(f"la derivee partielle de L par rapport a A3:\n{dA3} ")
#derivee partielle de Z3
dZ3=dA4*Relu_derivee(Z3)
print(f"derivee partielle par rapport a Z3:{dZ3}")
#derivee partielle par rappot a W3
dW3=np.dot(dZ3,A2.T)
print(f"derivee partielle par rapport a W3:{dW3}")
#derivee partielle par rapport a b3
db3=np.sum(dZ3,keepdims=True,axis=1)
print(f"la derivée partielle par rapport a b3 est :{db3}")
#derivée partielle par rapport a A2
dA2=np.dot(W3.T,dZ3)
print(f"la derivee partielle par rapport a A2 est :{dA2}")
#derivee partielle par rapport Z2
dZ2=dA2*Relu_derivee(Z2)
print(f"derivee partielle a Z2 est :{dZ2}")
#derivee partielle par rapport a W2
dW2=np.dot(dZ2,A1.T)
print(f"derivee partielle par rapport a W2 est :{dW2}")
#derivee partielle par rapport a b2
db2=np.sum(dZ2,axis=1,keepdims=True)
print(f"derivee partielle par rapport a b2 est :{db2}")
#derivee partielle par rapport a A1
dA1=np.dot(W2.T,dZ2)
print(f"derivee partielle par rapport a A1 est :{dA1}")
#derivee partielle par rapport a Z1
dZ1=dA1*Relu_derivee(Z1)
print(f" la derivee partielle par rapport a Z1 est :{dZ1}")
#derivee partielle par rapport a W1
dW1=np.dot(dZ1,X.T)
print(f"derivee partielle par rapport a W1 est :{dW1}")
#derivee partielle par rapport a b1
db1=np.sum(dZ1,axis=1,keepdims=True)
print(f"la derivve partielle par rapport a b1 est :{db1}")
#la mise a jour des parametres pour ajuste le model
learning_rate=.01 #taux d'apprentissage 
W1=W1-(learning_rate*dW1) #ici on calcul le nouveau W1
b1=b1- (learning_rate*db1)
W2=W2-(learning_rate*dW2)
b2=b2-(learning_rate*db2)
W3=W3-(learning_rate*dW3)
b3=b3-(learning_rate*db3)
W4=W4-(learning_rate*dW4)
b4=b4-(learning_rate*db4)
#la boucle d'entrainement avec un nombre de passage sur tout les donne ,epoche=10
epoche=10
for i in range(epoche):
  #1forward propagation
  Z1=np.dot(W1,X)+b1
  A1=Relu(Z1)
  Z2=np.dot(W2,A1)+b2
  A2=Relu(Z2)
  Z3=np.dot(W3,A2)+b3
  A3=Relu(Z3)
  Z4=np.dot(W4,A3)+b4
  A4=Z4 #sans fonction d'activation
  #2loss ou cost
  print(f"la prediction des reseaux est :{A4}")
  cost=MSE(Y_true,A4)
  print(f"le computer_cost est :{cost}")
  #une condition d'arret pour eviter des calculs inutile ,ou quand le loss bouche peu
  if cost<1e-6:
    print(f"convergence atteinte àl'epoche:{i+1}")
    break;
  #3Backpropagation:
  dA4 = MSE_derivee(Y_true, A4)

  dZ4 = dA4
  dW4 = np.dot(dZ4, A3.T)
  db4 = np.sum(dZ4, axis=1, keepdims=True)

  dA3 = np.dot(W4.T, dZ4)
  dZ3 = dA3 * Relu_derivee(Z3)
  dW3 = np.dot(dZ3, A2.T)
  db3 = np.sum(dZ3, axis=1, keepdims=True)

  dA2 = np.dot(W3.T, dZ3)
  dZ2 = dA2 * Relu_derivee(Z2)
  dW2 = np.dot(dZ2, A1.T)
  db2 = np.sum(dZ2, axis=1, keepdims=True)

  dA1 = np.dot(W2.T, dZ2)
  dZ1 = dA1 * Relu_derivee(Z1)
  dW1 = np.dot(dZ1, X.T)
  db1 = np.sum(dZ1, axis=1, keepdims=True)

    # 4. Mise à jour des paramètres
  W1 = W1 - learning_rate * dW1
  b1 = b1 - learning_rate * db1

  W2 = W2 - learning_rate * dW2
  b2 = b2 - learning_rate * db2

  W3 = W3 - learning_rate * dW3
  b3 = b3 - learning_rate * db3

  W4 = W4 - learning_rate * dW4
  b4 = b4 - learning_rate * db4
  
 


