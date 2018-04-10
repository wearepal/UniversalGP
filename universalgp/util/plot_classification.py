#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday March 29 12:41:54 2018

Usage: plot for classification, including sensitive attributes plot
"""
import matplotlib.pyplot as plt  # for plotting stuff
import numpy as np

MARKER = ['x', '+', 'o', 'x', 'v', '^', '<', '>', 's', 'p', 'h', 'd', 'H', 'D', 'X']
FILLED_MARKER = ['P', 'X', 'D', 'H', 's', '8','p']
EMPTY_MARKER = ['+', 'x', 'd', 'h']
COLOR = ['b', 'r', 'g', 'c', 'm', 'y', 'k']


class Classification:
    """Make a class for classification plot."""

    def __init__(self, inputs, labels):
        self.X = inputs
        self.y = np.squeeze(labels)
        self.N = len(self.y)

    def plot_2d(self, num_to_draw=200):
        if self.N < num_to_draw:
            x_draw = self.X
            y_draw = self.y
        else:
            x_draw = self.X[:num_to_draw]
            y_draw = self.y[:num_to_draw]

        plt.figure()
        plt.scatter(x_draw[:, 0], x_draw[:, 1], 20, label=y_draw)

    def plot_2d_sensitive(self, sensi_attri, num_to_draw=200):
        sensi_attri = np.squeeze(sensi_attri)

        if self.N < num_to_draw:
            x_draw = self.X
            y_draw = self.y
            s_draw = sensi_attri
        else:
            x_draw = self.X[:num_to_draw]
            y_draw = self.y[:num_to_draw]
            s_draw = sensi_attri[:num_to_draw]

        class_label = list(set(y_draw))
        sensitive_label = list(set(s_draw))

        plt.figure()
        for i in sensitive_label:
            X_s = x_draw[s_draw == i]
            y_s = y_draw[s_draw == i]
            sensitive_name = " ".join(["Sensi-Label=", str(int(i)), ","])

            for j in class_label:
                class_name = " ".join(["Class-Label=", str(int(j))])
                label = sensitive_name + class_name
                plt.scatter(X_s[y_s == j][:, 0], X_s[y_s == j][:, 1], color=COLOR[int(j)],
                            marker=MARKER[int(i)], label=label)

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc="best")
        plt.show()

    def plot_2d_sensitive_prediction(self, pred, sensi_attri, num_to_draw=200):
        pred = np.squeeze(pred)
        sensi_attri = np.squeeze(sensi_attri)
        false_pred_rate = self.false_rate(pred, sensi_attr_test)

        if self.N < num_to_draw:
            x_draw = self.X
            y_draw = self.y
            s_draw = sensi_attri
            pred_draw = pred
        else:
            x_draw = self.X[:num_to_draw]
            y_draw = self.y[:num_to_draw]
            s_draw = sensi_attri[:num_to_draw]
            pred_draw = pred[:num_to_draw]

        class_label = list(set(y_draw))
        sensitive_label = list(set(s_draw))

        plt.figure()
        for i in sensitive_label:
            X_s = x_draw[s_draw == i]
            y_s = y_draw[s_draw == i]
            pred_s = pred_draw[s_draw == i]

            sensitive_name = " ".join(["S-", str(int(i)), ","])

            for j in class_label:
                class_name = " ".join(["C-", str(int(j))])
                false_rate = round(false_pred_rate[int(i), int(j)], 3)
                true_rate = 1 - false_rate

                label_true = sensitive_name + class_name + ", TR=" + str(true_rate)
                same_insection = np.all([y_s == j, pred_s == j], axis=0)
                plt.scatter(X_s[same_insection][:, 0], X_s[same_insection][:, 1],
                            color=COLOR[int(j)], marker=FILLED_MARKER[int(i)], label=label_true, alpha=0.6)

                label_false = sensitive_name + class_name + ", FR=" + str(false_rate)
                diff_insection = np.all([y_s == j, pred_s != j], axis=0)
                plt.scatter(X_s[diff_insection][:, 0], X_s[diff_insection][:, 1], s= 50,
                            color=COLOR[int(j)], marker=EMPTY_MARKER[int(i)], label=label_false, alpha=0.6)

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc="best")
        plt.show()

    def false_rate(self, pred, sensi_attri):
        pred = np.squeeze(pred)
        sensi_attri = np.squeeze(sensi_attri)

        class_label = list(set(self.y))
        sensitive_label = list(set(sensi_attri))

        false_pred_rate = np.zeros((len(sensitive_label), len(class_label)))

        for i in sensitive_label:
            X_s = self.X[sensi_attri == i]
            y_s = self.y[sensi_attri == i]
            pred_s = pred[sensi_attri == i]
            sensitive_group = " ".join(["Sensi-Label=", str(int(i)), ","])
            for j in class_label:
                insection = np.all([y_s == j, pred_s != j], axis=0)
                false_pred_rate[int(i), int(j)] = len(X_s[insection][:, 0]) / len(X_s[:, 0])
                class_group = " ".join(["Class-Label=", str(int(j))])
                label = sensitive_group + class_group
                print(label + ', False prediction rate ' + str(false_pred_rate[int(i), int(j)]))

        return false_pred_rate






