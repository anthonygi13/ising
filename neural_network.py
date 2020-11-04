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


def load_network(path, n):
    W = []
    for l in range(1, n+1):
        W += [np.array(numpy.loadtxt(path+f"W{l}", ndmin=2))]
        print(W[-1].shape)
    return NeuralNetwork(W)



def initialise_W(M, training_set, renormalisation):
    assert M[0] == training_set.shape[1]
    assert M[-1] == 1
    x = training_set.T
    W = []
    for i in range(1, len(M)):
        W_T = np.array(numpy.random.uniform(size=(M[i], M[i-1])))
        z_T = W_T @ x
        mean = np.mean(z_T, axis=1)
        std = np.std(z_T, axis=1)
        W += [np.append(W_T, np.expand_dims(-mean, axis=1), axis=1) / (np.expand_dims(std, axis=1) * renormalisation)]
        x = sigma(W[-1] @ np.append(x, np.ones((1, x.shape[1])), axis=0))
    return W


class NeuralNetwork():
    def __init__(self, initial_W):
        """
        :param initial_W: list of n arrays W_l (l=1->n) with shape (M_l, M_{l-1}+1)
        and W_n must have shape (1, M_{n-1}+1)
        """
        assert training_set.shape[0] == labels.shape[0]
        M0 = training_set.shape[1]
        self.check_W(initial_W, M0)
        self.W = deepcopy(initial_W)
        self.n = len(initial_W)

    @staticmethod
    def check_W(W, M0):
        assert W[0].ndim == 2
        assert W[0].shape[1] == M0 + 1
        for lm1, Wl in enumerate(W[1:]):
            assert Wl.ndim == 2
            assert Wl.shape[1] == W[lm1].shape[0] + 1
        assert W[-1].shape[0] == 1

    @staticmethod
    def classification_func(input_vectors, *args):  # TODO: changer nom pour predicted proba ou un truc comme ca ?
        """
        :param input_vectors:
        :param args: W_1, W_2, ..., W_n
        :return: predicted probability for the input vector to correspond to a configuration at T>Tc
        """
        # TODO: tester avec un seul input_vector
        M0 = input_vectors.shape[1] if input_vectors.ndim == 2 else input_vectors.shape[0]
        NeuralNetwork.check_W(args, M0)
        x = input_vectors.T
        for W_l in args:
            xp = np.append(x, np.ones((1, x.shape[1])), axis=0)
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
        M0 = training_set.shape[1]
        NeuralNetwork.check_W(args, M0)
        prediction = NeuralNetwork.classification_func(training_set, *args)

        return np.mean(- (labels * np.log(prediction) + (1 - labels) * np.log(1 - prediction)))

    def get_cross_entropy(self, training_set, labels):
        """
        :param training_set: array of training vectors with shape (#samples, M_0)
        :param labels: 1D array of labels associated to the training vectors (either 0 or 1)
        :return:
        """
        return self.cross_entropy_func(training_set, labels, *self.W)

    def get_classification(self, input_vectors):
        """
        :param input_vectors: array of input vectors with shape (#vectors, M_0)
        :return:
        """
        return self.classification_func(input_vectors, *self.W)

    def get_grad(self, training_set, labels):
        """
        :param training_set: array of training vectors with shape (#samples, M_0)
        :param labels: 1D array of labels associated to the training vectors (either 0 or 1)
        :return:
        """
        return grad(NeuralNetwork.cross_entropy_func, argnums=np.arange(2, self.n+2))(training_set, labels, *self.W)
        #return [grad(NeuralNetwork.cross_entropy_func, argnums=i+2)(self.training_set, self.labels, *self.W) for i in range(self.n)]

    def training(self, training_set, labels, step_size, epsilon=None, nsteps=None):
        if epsilon is None and nsteps is None:
            raise ValueError("At least one parameter must be filled among epsilon and nsteps.")

        cross_entropies = [self.get_cross_entropy(training_set, labels)]
        validity_rates = [self.get_validity_rate(training_set, labels)]
        i = 0
        while True:
            try:
                grad_W = self.get_grad(training_set, labels)
                for l in range(self.n):
                    self.W[l] -= step_size * grad_W[l]
                cross_entropies += [self.get_cross_entropy(training_set, labels)]
                validity_rates += [self.get_validity_rate(training_set, labels)]
                if i % 10 == 0:
                    print(f"Training step {i}/{nsteps}")
                i += 1
                if nsteps is not None and i > nsteps:
                    break
                if epsilon is not None and cross_entropies[-2] - cross_entropies[-1] < epsilon:
                    break
            except KeyboardInterrupt:
                break
        return cross_entropies, validity_rates

    def save_network(self, path):
        for l, Wl in enumerate(self.W):
            numpy.savetxt(path + f"W{l+1}", Wl)

    def get_validity_rate(self, vectors, labels):
        predicted_proba = self.get_classification(vectors)
        return np.mean((predicted_proba > 0.5) == labels)


# Main
if __name__ == "__main__":
    dir = "set1/"
    path = "nw1/"
    nsteps = 10000
    step_size = 0.01
    renormalisation = 3

    training_set = np.array(numpy.loadtxt(dir + "vectors"))
    labels = np.array(numpy.loadtxt(dir + "classification"))
    M0 = training_set.shape[1]
    M = [M0, 2000, 1]

    W = initialise_W(M, training_set, renormalisation)
    nw = NeuralNetwork(W)
    #nw = load_network(path, 2)

    cross_entropies, validity_rates = nw.training(training_set, labels, step_size, nsteps=nsteps)

    nw.save_network(path)

    plt.subplot(121)
    plt.plot(np.arange(1, len(cross_entropies)+1), cross_entropies, label="cross entropy")
    plt.xlabel("Epoch")
    plt.subplot(122)
    plt.plot(np.arange(1, len(validity_rates)+1), validity_rates, label="validity_rates")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
