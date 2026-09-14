#impementation manuelle d'un reseau profond 
#avec les technique d'initiation en fonction de la fonction d'activation
#there , we work with relu for the forward pass 
#My english is so bad🤣
from matplotlib import pyplot as plt
import sys as sys
#import des fonctions d'activation
sys.path.append("../../01-fondamentaux/04-fonctions-activation")
from relu import Relu ,Relu_derivee
#import de la fonction de perte MSE pour la regression 
sys.path.append("../01-fonction-perte")
from mse import  MSE,MSE_derivee
import numpy as np
#implementation de la classe neurone
class Reseaux:
  def __init__(self,architecture,learning_rate=0.01,nb_epoque=100):
    self.architecture=architecture
    self.nb_couches=len(self.architecture)-1
    self.eta=learning_rate
    self.nb_epoques=nb_epoque
    #un dictionnaire de parametres pour les sorties lors du forwardpass
    self.parametres={}
    #initiation des matrices de poids et biais avec l'initialisation de He pour relu
    for i in range(1,self.nb_couches+1):
      init_he=np.sqrt(2/(self.architecture[i-1]))
      self.parametres[f"W{i}"]=np.random.randn(self.architecture[i],self.architecture[i-1])*init_he
      self.parametres[f"b{i}"]=np.zeros((self.architecture[i],1))


  #======================================#
  #la propation avant du reseaux
  def forward_passe(self,X):
    #utilisation d'un dictionnaire pour les gradients de parametres
     A=X
     caches={"A0":X}
     for i in range(1,self.nb_couches+1):
      #comme c'est la regression on applique pas un fonction d'activation sur la sortie finale(la prediction)
        Z=np.dot(self.parametres[f"W{i}"],A)+self.parametres[f"b{i}"]
        caches[f"Z{i}"]=Z #stockage de chaque d'une couche du reseau
        #application de relu pour les couches caches
        if i<self.nb_couches:
           A=Relu(Z)
           
        else:
            A=Z
        caches[f"A{i}"]=A
     return A,caches
  #==============================================#
  #la fonction de perte ou le coût
  def cost(self,y,y_predict):
     return MSE(y,y_predict)
  #==============================================#
  #la retropopagation ,propagation arriere
  def grandient_parametres(self,X,y):
      y_predict,caches=self.forward_passe(X)
      gradients={} #le gradient des parametres pour la mis a jours
      dJ=MSE_derivee(y,y_predict)
      dA=dJ
      gradients[f"dZ"]=dJ
      for i in range(self.nb_couches,0,-1):
          if i==self.nb_couches:
              #ici comme c'est un regression et pas de fonction d'activation sur la sorte de sortie donc dA_finale=dZ
              dZ=dA # couche de sortie ,sortie linaire
          else:
              dZ=dA*Relu_derivee(caches[f"Z{i}"])
          #calculs des derivees partielles par rapport a chaque parametres
          gradients[f"W{i}"]=np.dot(dZ,caches[f"A{i-1}"].T)
          gradients[f"b{i}"]=np.sum(dZ,axis=1,keepdims=True)
        #gradient pour la couche précedente
          if i>1:
              dA=np.dot(self.parametres[f"W{i}"].T,dZ)
      return gradients
  
  #==============================================#
  # la descente de gradient avec mis ajours
  def mise_a_jour(self,gradient):
          for i in range(1,self.nb_couches+1):
             self.parametres[f"W{i}"]-=self.eta*gradient[f"W{i}"]
             self.parametres[f"b{i}"]-=self.eta*gradient[f"b{i}"]
        

 
  #==============================================#
  #entrainement du  reseau de neurone
  def fit(self,X,y):
    historique_error=[]
    for epoque in range(self.nb_epoques):
        A_finale,caches=self.forward_passe(X)
        costs=self.cost(y=y,y_predict=A_finale)
        historique_error.append(costs)
        gradients=self.grandient_parametres(X,y)
        self.mise_a_jour(gradient=gradients)
    return historique_error
  #==============================================#
    #prediction du model
  def predict(self,X):
          A,_=self.forward_passe(X)
          return A


#teste du reseau de neurone
X=np.array([[1,4,7],[2,5,8],[3,6,9]])
print("X :")
print(X)
print("shape X :", X.shape)
y=np.array([[10,20,30]])
print("y :")
print(y)
print("shape y :", y.shape)
model=Reseaux(architecture=[3,5,3,1],learning_rate=0.01,nb_epoque=100)
historique=model.fit(X,y)
prediction=model.predict(X)
precision=np.mean(model.predict==y)
print(f"la prediction du model est :{prediction}")
print(f"la precision du model est :{precision}")
print(f"la loss initiale est :{historique[0]}")
print(f"la loss finale est :{historique[-1]}")
plt.figure(figsize=(10,5))
plt.plot(historique)
plt.title("variation du coup pendant l'entrainement")
plt.xlabel("epoque")
plt.ylabel("cout")
plt.show()




      
     
      

        

     
     


   




