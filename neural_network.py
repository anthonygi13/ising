#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : neural_network.py
# Created by Anthony Giraudo and Clement Sebastiao the 30/10/2020

# Modules

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from jax import grad
import jax.numpy as jnp


# Functions

def sigma(x, z):
    return 1/(1+jnp.exp(-(z+z)))


class NeuralNetwork():
    def __init__(self, initial_W, initial_p, training_set, labels):
        """
        :param initial_W: list of n arrays W_l (l=1->n) with shape (M_l, M_{l-1}+1)
        :param initial_p: vector of shape (M_n,)
        :param training_set: array of training vectors with shape (#samples, M_0)
        :param labels: 1D array of labels associated to the training vectors (either 0 or 1)
        """
        self.training_set = np.array(training_set, copy=True)
        self.labels = np.array(labels, copy=True)
        assert training_set.shape[0] == labels.shape[0]
        self.M = [training_set.shape[1]]  # list of n+1 M_l (l=0->n)
        print("M0:", self.M[-1])
        self.check_p_W(initial_p, initial_W, self.M[0])
        for W_l in initial_W:
            self.M += [W_l.shape[0]]
        self.W = deepcopy(initial_W)
        self.p = np.expand_dims(initial_p, axis=0)  # shape (M_n,)
        self.n = len(initial_W)

        self.cross_entropy = np.empty((0,))  # shape (#opt_step,)

    @staticmethod
    def check_p_W(p, W, M0):
        assert W[0].ndim == 2
        assert W[0].shape[1] == M0 + 1
        assert p.ndim == 1
        assert p.shape[0] == W[-1].shape[0]
        for lm1, Wl in enumerate(W[1:]):
            assert Wl.ndim == 2
            assert Wl.shape[1] == W[lm1].shape[0] + 1

    def calculate_cross_entropy(self):
        pass  # TODO: gerer tous les samples en meme temps

    def classify(self, input_vector):
        pass

    def training(self, epsilon=None, nsteps=None):
        pass  # TODO: arret manuel, choix entre nsteps et epsilon, et si les 2 ca fait epsilon avec un nsteps max
        # TODO: ou utiliser une librairie ?


# Main
if __name__ == "__main__":
    derivative_fn = grad(sigma)
    print(derivative_fn(0.))

