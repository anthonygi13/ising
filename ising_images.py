# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

#Functions
"Initialise un réseau carré (L,L) de spins."
def init_lattice(L=64):
    return 2*np.random.randint(2,size=(L,L))-1

"Calcule l'énergie de notre réseau de spins."
def energy(latt,J=1):
    E=0
    L=len(latt)
    for i in range(L-1):
        for j in range(L-1):
            E+=J*(latt[i][j]*latt[i][j+1]+latt[i][j]*latt[i+1][j])
    return -E

"Calcule la magnétisation de notre réseau."
def mag(latt):
    return np.sum(latt)
    
    
"Méthode de Monte-Carlo."
def MC(latt,N=500000,T=0.5):
    L=len(latt)
    for i in range(N):
        "On choisit un état au hasard"
        a=np.random.randint(L)
        b=np.random.randint(L)
        s=latt[a][b]
        neighbors=latt[(a+1)%L,b]+latt[a,(b+1)%L]+latt[(a-1)%L,b]+latt[a,(b-1)%L]
        "On compare l'énergie des 2 réseaux"
        dE=2*s*neighbors
        """
        Si l'énergie est plus faible, on
        fait le changement sinon on le
        fait suivant une certaine 
        probabilité seulement
        """
        if dE<0:
            latt[a][b]=-s
        else:
            nu=np.random.uniform(0,1)
            if nu<np.exp(-(dE)/T):
                latt[a][b]=-s
    return latt
    

#Main
for j in range(20):
    T=0.6+j*0.2
    latt=init_lattice()
    nlatt=MC(latt,N=2000000,T=T)
    for i in range(10):
        nlatt=MC(nlatt,T=T)
        plt.imsave("./images/image_{}_{}.png".format(round(T,1),i),nlatt)
