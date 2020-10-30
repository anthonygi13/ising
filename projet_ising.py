# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

#Functions
"Initialise un réseau carré (L,L) de spins."
def init_lattice(L=32):
    return 2*np.random.randint(2,size=(L,L))-1

"Calcule l'énergie de notre réseau de spins."
def energy(latt):
    E=0
    L=len(latt)
    for i in range(L-1):
        for j in range(L-1):
            E+=(latt[i][j]*latt[i][j+1]+latt[i][j]*latt[i+1][j])
    return -E

"Calcule la magnétisation de notre réseau."
def mag(latt):
    return np.sum(latt)
    
    
"Méthode de Monte-Carlo."
def MC(latt,N=10000,T=0.5):
    L=len(latt)
    for i in range(N):
        "On choisit un état au hasard"
        a=np.random.randint(L)
        b=np.random.randint(L)
        """
        On fait un nouveau réseau avec
        l'état choisi qui est inversé
        """
        nlatt=np.copy(latt)
        nlatt[a][b]=-latt[a][b]
        "On compare l'énergie des 2 réseaux"
        old_energy=energy(latt)
        new_energy=energy(nlatt)
        """
        Si l'énergie est plus faible, on
        fait le changement sinon on le
        fait suivant une certaine 
        probabilité seulement
        """
        if new_energy<old_energy:
            latt=nlatt
        else:
            nu=np.random.uniform(0,1)
            if nu<np.exp(-(new_energy-old_energy)/T):
                latt=nlatt
    return latt
    

#Main
latt=init_lattice()
E=energy(latt)
M=mag(latt)
print("Réseau initial d'énergie E={} et de magnétisation M={}".format(E,M))
plt.figure()
plt.imshow(latt,'binary')

nlatt=MC(latt)
nE=energy(nlatt)
nM=mag(nlatt)
print("Réseau final d'énergie E={} et de magnétisation M={}".format(nE,nM))
plt.figure()
plt.imshow(nlatt,'binary')
