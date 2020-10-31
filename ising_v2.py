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
    L = latt.shape[0]

    # choisit un spin random
    i, j = np.random.randint(L, size=(2,))

    # calcul ecart d'energie si spin change
    s = latt[i, j]
    neighbors = latt[(i+1) % L, j] + latt[i, (j+1) % L] + latt[(i-1) % L, j]+latt[i, (j-1) % L]
    dE = 2*s*neighbors

    # acceptation/rejet
    if dE <= 0:
        latt[i, j] *= -1
    elif np.random.uniform() < np.exp(-dE/T):
        latt[i, j] *= -1


def mc(latt, N_eq, N, T, separation):
    nlatt = np.array(latt, copy=True)
    for i in range(N_eq):
        if i%10000 == 0:
            print(f"Equilibrage: {i}/{N_eq}")
        mc_step(nlatt, T)

    sampled_latt = [np.array(nlatt, copy=True)]
    for i in range(N):
        if i % 10000 == 0:
            print(f"MC step : {i}/{N}")
        mc_step(nlatt, T)
        if (i+1) % separation == 0:
            sampled_latt += [np.array(nlatt, copy=True)]

    return np.array(sampled_latt)  # shape (#latt, L, L)


# Main

if __name__ == "__main__":
    dir = "set1/"
    L = 64
    N_eq = int(2e6)
    N = int(25e5)
    separation = int(5e5)
    list_T = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1]
    nb_mc = 10

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
        for mc_id in range(nb_mc):
            print(f"Avancement : T={T}, mc_id={mc_id}/{nb_mc}")

            subdir = dir + f"T={T}_mc_id={mc_id}/"
            os.mkdir(subdir)

            latt = init_lattice(L)
            sampled_latt = mc(latt, N_eq, N, T, separation)

            nsamples = sampled_latt.shape[0]
            vectors += np.split(sampled_latt.flatten(), nsamples)
            classification += [0 if T < critical_temperature() else 1]*nsamples

            for i, sample in enumerate(sampled_latt):
                plt.imsave(subdir+f"{i}.png", sample)

    np.savetxt(dir+"vectors", np.array(vectors), fmt="%d")  # shape (#samples, L**2)
    np.savetxt(dir+"classification", np.array(classification), fmt="%d")  # shape (#samples,)










