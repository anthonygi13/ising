# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

#Functions
"Initialise un réseau carré (L,L) de spins."
def init_lattice(L=100):
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
def MC_step(latt,T=0.5):
    L=len(latt)
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
    

def animate(i,plot):
    global latt
    latt=MC_step(latt)
    plot.set_array(latt)
    return plot,

latt=init_lattice()
fig = plt.figure()
plot = plt.imshow(latt, 'afmhot')
plt.axis('off')
anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=10000, interval=1,
                                       fargs=(plot,), repeat=False)
plt.show()
