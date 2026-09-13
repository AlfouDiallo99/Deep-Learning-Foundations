#dans cette partie ,on prevoit de voir comment les poids sont initialiser
#vue que les poids du model ne sont pas connus l'avence , on utilise generalement des poids aleatoirs  avec np.random.randn(),la loi normale
#et comme  ses poids sont aleatoir , elles peuvent etre 0 ,1 ou explosé 
#donc pour remedier au probleme on utilise des facteurs pour que les soit tres poche de Zero facilitant ainsi les calculs 
#les methoe utilisées pour le calculs de facteurs sont He pour Relu ou ses variantes
#Xavier / Glorot pour les fonctions d'activation de Sigmoïd ou Tanh
import numpy as np
#le shape des inputs(les entrees)
"""n_in,n_out=4,3
#initiasation avec Xavier/Glorot
Xavier_Glorot=np.sqrt(2/(n_in+n_out))
#le generation aleatoire des poids 
W_Xavier_Glorot=np.random.randn(n_out,n_in)*Xavier_Glorot
#le biais ,generalement a zero au depart 
b_Xavier_Glorot=np.zeros((n_out,1))
print(W_Xavier_Glorot)
print(f"Shape de W:{W_Xavier_Glorot.shape}")
print(f"Shape de b:{b_Xavier_Glorot.shape}")
print("=="*20)
#initialisation avec He
n_in,n_out=4,3
He=np.sqrt(2/n_in)
W_he=np.random.randn(n_out,n_in)*He
b_he=np.zeros((n_out,1))
print(f"W_He:\n{W_he}")
print(f"shape de W_He:{W_he.shape}")
print(f"b_He:{b_he}")
print(f"shape de b_He:{b_he.shape}")"""
# Xavier
n_in,n_out=4,3
Xavier_Glorot=np.sqrt(2/(n_in+n_out))
W_xavier =np.random.randn(1000)*Xavier_Glorot
He=np.sqrt(2/n_in)
# He
W_he =np.random.randn(1000)*He
print("Moyenne Xavier :",W_xavier.mean())
print("Ecart-type Xavier :",W_xavier.std())
print("Moyenne He :",W_he.mean())
print("Ecart-type He :",W_he.std())
#import de la fonction relu
import sys as sys
sys.path.append("../../01-fondamentaux/04-fonctions-activation")
from relu import Relu as rl
Z=np.random.randn(1000)
A=rl(Z)
number_zero=np.sum(A==0)
print(f"nombre de valeurs nulles est :{number_zero}")
print(f"la proportion de valeurs nulle :{number_zero/len(A)*100}")
#print(A)
Z=np.random.randn(1000)
Z_xavier = np.random.randn(1000) * Xavier_Glorot
Z_he = np.random.randn(1000) * He
A_xavier = rl(Z_xavier)
A_he = rl(Z_he)
number_nul_xavire=np.sum(A_xavier==0)
number_nul_he=np.sum(A_he==0)
print(f"le nombre de valeur nuls de Xavier est :{number_nul_xavire}")
print(f" la  proportion de valeurs nul de Xavier est :{number_nul_xavire/len(A_xavier)*100}%")
print(f"le nombre de valeur nuls de he est :{number_nul_he}")
print(f" la  proportion de valeurs nul de He est :{number_nul_he/len(A_he)*100}%")
print("Moyenne activation Xavier :", A_xavier.mean())
print("Moyenne activation He :", A_he.mean())
print("Ecart-type activation Xavier :", A_xavier.std())
print("Ecart-type activation He :", A_he.std())
#creation de X avec des valeurs aléatoir juste pour experiementé Xavier/Glorot et He
m=100# le nombre d'exemple
X_input=np.random.randn(n_in,m)
print(X_input.shape)
sigma_xavier=np.sqrt(2/(n_in+n_out))
W_xavier=np.random.randn(n_out,n_in)*sigma_xavier
b_xavier=np.zeros((n_out,1))
Z_xavier=np.dot(W_xavier,X_input)+b_xavier
A_xavier=rl(Z_xavier)
print(A_xavier.shape)
print(np.mean(A_xavier))
print(np.std(A_xavier))
print(np.sum(A_xavier == 0)*100,f"%")