#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : neural_network.py
# Created by Anthony Giraudo and Clement Sebastiao the 30/10/2020

# Modules

import matplotlib.pyplot as plt
from copy import deepcopy
from jax import grad
import jax.numpy as np
import numpy


# Functions

def sigma(x):
    return 1/(1+np.exp(-x))


def load_network():
    pass  # TODO, et reflechir a si c vraiment pratique d avoir training_set te classification en attributs


class NeuralNetwork():
    def __init__(self, initial_W, training_set, labels):
        """
        :param initial_W: list of n arrays W_l (l=1->n) with shape (M_l, M_{l-1}+1)
        and W_n must have shape (1, M_{n-1}+1)
        :param training_set: array of training vectors with shape (#samples, M_0)
        :param labels: 1D array of labels associated to the training vectors (either 0 or 1)
        """
        self.training_set = np.array(training_set, copy=True)
        self.labels = np.array(labels, copy=True)
        assert training_set.shape[0] == labels.shape[0]
        self.M = [training_set.shape[1]]  # list of n+1 M_l (l=0->n)
        print("M0:", self.M[-1])
        self.check_W(initial_W, self.M[0])
        for W_l in initial_W:
            self.M += [W_l.shape[0]]
        self.W = deepcopy(initial_W)
        self.n = len(initial_W)

        self.cross_entropies = []

    @staticmethod
    def check_W(W, M0):
        assert W[0].ndim == 2
        assert W[0].shape[1] == M0 + 1
        for lm1, Wl in enumerate(W[1:]):
            assert Wl.ndim == 2
            assert Wl.shape[1] == W[lm1].shape[0] + 1
        assert W[-1].shape[0] == 1

    @staticmethod
    def classification_func(input_vectors, *args):
        """
        :param input_vectors:
        :param args: W_1, W_2, ..., W_n
        :return: predicted probability for the input vector to correspond to a configuration at T>Tc
        """
        M0 = input_vectors.shape[1] if input_vectors.ndim == 2 else input_vectors.shape[0]
        NeuralNetwork.check_W(args, M0)
        x = input_vectors.T
        for W_l in args:
            xp = np.append(x, np.ones((1, x.shape[1])), axis=0)  # TODO: tester avec un seul input_vector
            z = W_l @ xp
            x = sigma(z)
        return x[0]

    @staticmethod
    def cross_entropy_func(training_set, labels, *args):
        """
        Cross entropy function for automatic differentation
        :param args: W_1, W_2, ..., W_n
        :return:
        """
        # TODO: checker formule
        M0 = training_set.shape[1]
        NeuralNetwork.check_W(args, M0)
        prediction = NeuralNetwork.classification_func(training_set, *args)

        return np.sum(- (labels * np.log(prediction) + (1 - labels) * np.log(1 - prediction)))  # FIXME: log(0)=-infty...

    def get_cross_entropy(self):
        return self.cross_entropy_func(self.training_set, self.labels, *self.W)

    def get_classification(self, input_vectors):
        return self.classification_func(input_vectors, *self.W)

    def get_grad(self):
        return grad(NeuralNetwork.cross_entropy_func, argnums=np.arange(2, self.n+2))(self.training_set, self.labels, *self.W)
        #return [grad(NeuralNetwork.cross_entropy_func, argnums=i+2)(self.training_set, self.labels, *self.W) for i in range(self.n)]

    def training(self, step_size, epsilon=None, nsteps=None):
        if epsilon is None and nsteps is None:
            raise ValueError("At least one parameter must be filled among epsilon and nsteps.")

        self.cross_entropies += [self.get_cross_entropy()]
        i = 0
        while True:
            grad_W = self.get_grad()
            for l in range(self.n):
                self.W[l] -= step_size * grad_W[l]
            self.cross_entropies += [self.get_cross_entropy()]

            i += 1
            if nsteps is not None and i > nsteps:
                break
            if epsilon is not None and self.cross_entropies[-2] - self.cross_entropies[-1] < epsilon:
                break

    def save_network(self, path):
        pass  # TODO


# Main
if __name__ == "__main__":
    dir = "set1/"
    training_set = np.array(numpy.loadtxt(dir+"vectors"))
    labels = np.array(numpy.loadtxt(dir+"classification"))

    # TODO: mieux choisir coefficients pour etre dans la partie lineaire du log : faire des calculs de moyenne et variance sur training set pour gerer ça
    M0 = training_set.shape[1]
    M = [M0, 2000, 1]
    W = []
    for i in range(1, len(M)):
        W += [np.array(numpy.random.uniform(size=(M[i], M[i-1]+1)))]

    nw = NeuralNetwork(W, training_set, labels)

    print(nw.training_set)
    print(nw.labels)
    print(nw.classification_func(nw.training_set, *nw.W))
    print(nw.get_classification(nw.training_set))
    print(nw.cross_entropy_func(training_set, labels, *nw.W))
    print(nw.get_cross_entropy())

    #nw.training(0.01, nsteps=10)
    plt.plot(np.arange(len(nw.cross_entropies)), nw.cross_entropies)
    plt.show()