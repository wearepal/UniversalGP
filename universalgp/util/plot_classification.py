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

        class_label = list(set(y_draw))

        plt.figure()
        for j in class_label:
            plt.scatter(x_draw[y_draw == j][:, 0], x_draw[y_draw == j][:, 1], color=COLOR[int(j)], label=str(int(j)))

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc="best")
        plt.show()

    def plot_2d_prediction(self, pred, num_to_draw=200):
        pred = np.squeeze(pred)
        false_pred_rate = self.false_rate(pred)

        if self.N < num_to_draw:
            x_draw = self.X
            y_draw = self.y
            pred_draw = pred
        else:
            x_draw = self.X[:num_to_draw]
            y_draw = self.y[:num_to_draw]
            pred_draw = pred[:num_to_draw]

        class_label = list(set(y_draw))

        plt.figure()
        for j in class_label:
            false_rate = round(false_pred_rate[int(j)], 2)
            true_rate = round(1 - false_rate, 2)

            true_pred = np.all([y_draw == j, pred_draw == j], axis=0)
            plt.scatter(x_draw[true_pred][:, 0], x_draw[true_pred][:, 1], marker='+',
                        color=COLOR[int(j)], label='C-' + str(int(j)) + ' T-rate=' + str(true_rate))

            false_pred = np.all([y_draw == j, pred_draw != j], axis=0)
            plt.scatter(x_draw[false_pred][:, 0], x_draw[false_pred][:, 1], marker='P',
                        color=COLOR[int(j)], label='C-' + str(int(j)) + ' F-rate=' + str(false_rate))

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc="best")
        plt.show()

    def plot_2d_sensitive(self, sensi_attr, num_to_draw=200):
        sensi_attr = np.squeeze(sensi_attr)

        if self.N < num_to_draw:
            x_draw = self.X
            y_draw = self.y
            s_draw = sensi_attr
        else:
            x_draw = self.X[:num_to_draw]
            y_draw = self.y[:num_to_draw]
            s_draw = sensi_attr[:num_to_draw]

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

    def plot_2d_sensitive_prediction(self, pred, sensi_attr, num_to_draw=200):
        pred = np.squeeze(pred)
        sensi_attr = np.squeeze(sensi_attr)
        false_pred_rate = self.false_rate(pred, sensi_attr)

        if self.N < num_to_draw:
            x_draw = self.X
            y_draw = self.y
            s_draw = sensi_attr
            pred_draw = pred
        else:
            x_draw = self.X[:num_to_draw]
            y_draw = self.y[:num_to_draw]
            s_draw = sensi_attr[:num_to_draw]
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
                false_rate = round(false_pred_rate[int(i), int(j)], 2)
                true_rate = round(1 - false_rate, 2)

                label_true = sensitive_name + class_name + ", TR=" + str(true_rate)
                true_pred = np.all([y_s == j, pred_s == j], axis=0)
                plt.scatter(X_s[true_pred][:, 0], X_s[true_pred][:, 1],
                            color=COLOR[int(j)], marker=FILLED_MARKER[int(i)], label=label_true, alpha=0.6)

                label_false = sensitive_name + class_name + ", FR=" + str(false_rate)
                false_pred = np.all([y_s == j, pred_s != j], axis=0)
                plt.scatter(X_s[false_pred][:, 0], X_s[false_pred][:, 1], s= 50,
                            color=COLOR[int(j)], marker=EMPTY_MARKER[int(i)], label=label_false, alpha=0.6)

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc="best")
        plt.show()

    def false_rate(self, pred, sensi_attr=None):
        pred = np.squeeze(pred)

        if sensi_attr is None:
            class_label = list(set(self.y))
            false_pred_rate = np.zeros(len(class_label))

            for j in class_label:
                false_pred = np.all([self.y == j, pred != j], axis=0)
                false_pred_rate[int(j)] = len(self.X[false_pred][:, 0]) / len(self.X[self.y == 1][:, 0])
                class_group = " ".join(["Class-Label=", str(int(j))])
                print(class_group + ', False prediction rate ' + str(false_pred_rate[int(j)]))

        else:
            sensi_attr = np.squeeze(sensi_attr)

            class_label = list(set(self.y))
            sensitive_label = list(set(sensi_attr))

            false_pred_rate = np.zeros((len(sensitive_label), len(class_label)))

            for i in sensitive_label:
                X_s = self.X[sensi_attr == i]
                y_s = self.y[sensi_attr == i]
                pred_s = pred[sensi_attr == i]
                sensitive_group = " ".join(["Sensitive-Label=", str(int(i)), ","])

                for j in class_label:
                    false_pred = np.all([y_s == j, pred_s != j], axis=0)
                    false_pred_rate[int(i), int(j)] = len(X_s[false_pred][:, 0]) / len(X_s[y_s == j][:, 0])
                    class_group = " ".join(["Class-Label=", str(int(j))])
                    label = sensitive_group + class_group
                    print(label + ', False prediction rate ' + str(false_pred_rate[int(i), int(j)]))

        return false_pred_rate






