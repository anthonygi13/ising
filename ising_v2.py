#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : mc_functions.py
# Created by Anthony Giraudo adn Clement Sebastiao the 31/10/2020

"""Nous travaillons avec le modele d'ising ou J et k_b ont ete pris egaux a 1
"""

# Modules

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation


# Functions

def critical_temperature():
    return 2/np.log(1+np.sqrt(2))


def init_lattice(L):
    """
    :param L: taille du reseau
    :return: initialise reseau carre de L*L spins
    """
    return 2*np.random.randint(2, size=(L, L)) - 1


def mc_step(latt, T):
    nlatt = np.array(latt, copy=True)
    L = latt.shape[0]

    # choisit un spin random
    i, j = np.random.randint(L, size=(2,))

    # calcul ecart d'energie si spin change
    s = latt[i, j]
    neighbors = latt[(i+1) % L, j] + latt[i, (j+1) % L] + latt[(i-1) % L, j]+latt[i, (j-1) % L]
    dE = 2*s*neighbors

    # acceptation/rejet
    if dE <= 0:
        nlatt[i, j] *= -1
    elif np.random.uniform() < np.exp(-dE/T):
        nlatt[i, j] *= -1

    return nlatt


def mc(latt, N_eq, N, T, separation):
    sampled_latt = [latt]
    nlatt = latt

    for i in range(N_eq):
        if i%10000 == 0:
            print(f"Equilibrage: {i}/{N_eq}")
        nlatt = mc_step(nlatt, T)

    for i in range(N):
        if i % 10000 == 0:
            print(f"MC step : {i}/{N}")
        if (i+1) % separation == 0:
            sampled_latt += [np.array(nlatt, copy=True)]
        nlatt = mc_step(nlatt, T)

    return np.array(sampled_latt)  # shape (N, L, L)


# Main

if __name__ == "__main__":
    dir = "test1/"
    L = 64
    N_eq = int(1e6)
    N = int(5e6)
    separation = int(1e6)
    list_T = [1.4, 1.5, 1.6, 1.7, 1.8, 2.6, 2.7, 2.8, 2.9, 3.0]

    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        raise ValueError(f"Directory {dir} already exists.")

    fichier = open(dir+"params.txt", "a")
    fichier.write(f"L={L}\nN_eq={N_eq}\nN={N}\nseparation={separation}\nlist_T={list_T}")
    fichier.close()

    vectors = []
    classification = []

    for T in list_T:
        subdir = dir + f"T={T}/"
        os.mkdir(subdir)

        latt = init_lattice(L)
        sampled_latt = mc(latt, N_eq, N, T, separation)

        nsamples = sampled_latt.shape[0]
        vectors += np.split(sampled_latt.flatten(), nsamples)
        classification += [0 if T < critical_temperature() else 1]*nsamples

        for i, sample in enumerate(sampled_latt):
            plt.imsave(subdir+f"{i}.png", sample)
        """
        fig = plt.figure()
        ims = []
        for sample in sampled_latt:
            im = plt.imshow(sample, animated=True)
        ani = animation.ArtistAnimation(fig, ims, interval=50)
        """
    np.savetxt(dir+"vectors", np.array(vectors), fmt="%d")
    np.savetxt(dir+"classification", np.array(classification), fmt="%d")









