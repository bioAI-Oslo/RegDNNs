"""
Code taken from:
https://github.com/Vemundss/unitary_memory_indexing/blob/main/src/AnimatedScatter.py
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from IPython.display import HTML


def projection_rejection(u, v):
    """
    projection of u on v, and rejection of u from v
    """
    proj = ((u @ v) / (v @ v)) * v
    reject = u - proj
    return proj, reject


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, cluster_data, weight_data, loss_history, n_clusters=4, acc=None):
        """
        Args:
            cluster_data: (N,2), where N=4*n -> four clusters
            weight_data: (#epochs,2,4)
            acc: end accuracy
        """
        self.cluster_data = cluster_data
        self.weight_data = weight_data
        self.loss_history = loss_history
        self.n_clusters = n_clusters
        self.acc = acc

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.

        self.save_count = weight_data.shape[0]
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.setup_plot,
            interval=25,  # time in ms between frames
            # repeat_delay=1000, # delay before loop
            blit=False,  # for OSX?
            save_count=self.save_count,  # #frames
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        data = self.cluster_data
        
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        #colors = ["red", "green", "blue", "orange", "purple"]
        self.means = np.mean(
            np.reshape(data, (self.n_clusters, int(data.shape[0] / self.n_clusters), data.shape[-1])), axis=1
        )
        color_idxs = np.argmax(
            self.means @ self.weight_data[0], axis=0
        )  # shapes (self.n_clusters,2) @ (2,self.n_clusters)
        cluster_N = int(data.shape[0] / self.n_clusters)
        self.weight_arrows = []
        self.rejection_arrows = []
        self.projection_arrows = []
        for color, mean, i in zip(colors, self.means, range(len(colors))):
            self.ax.scatter(
                data[i * cluster_N : (i + 1) * cluster_N, 0],
                data[i * cluster_N : (i + 1) * cluster_N, 1],
                color=color,
            )
            self.ax.arrow(
                0,
                0,
                *mean,
                length_includes_head=True,
                width=0.01,
                color=(0, 0, 0, 0.5),  # semi-transparent black arrow
            )

            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
            )

            proj, reject = projection_rejection(self.weight_data[0, :, i], mean)
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)
        self.ax.grid("on")
        self.ax.set_title("Loss={}".format(self.loss_history[0]))

    def update(self, k):
        """Update the scatter plot."""
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        color_idxs = np.argmax(
            self.means @ self.weight_data[k], axis=0
        )  # shapes (4,2) @ (2,4)
        for i in range(self.n_clusters):
            self.weight_arrows.pop(0).remove()  # delete arrow
            self.rejection_arrows.pop(0).remove()
            self.projection_arrows.pop(0).remove()
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],  # color=(1, 0, 0, 0.5),
            )

            proj, reject = projection_rejection(
                self.weight_data[k, :, i], self.means[i]
            )
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.5,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)

        self.ax.set_title(f"Loss={self.loss_history[k]} End accuracy={self.acc}")

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.weight_arrows

    def update(self, k):
        """Update the scatter plot."""
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        color_idxs = np.argmax(
            self.means @ self.weight_data[k], axis=0
        )  # shapes (4,2) @ (2,4)
        for i in range(4):
            self.weight_arrows.pop(0).remove()  # delete arrow
            self.rejection_arrows.pop(0).remove()
            self.projection_arrows.pop(0).remove()
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],  # color=(1, 0, 0, 0.5),
            )

            proj, reject = projection_rejection(
                self.weight_data[k, :, i], self.means[i]
            )
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.5,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)

        self.ax.set_title("Loss={}".format(self.loss_history[k]))

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.weight_arrows

def generate_data():
    size = (128, 2)
    N = size[0]
    first_quadrant = np.random.normal(loc = (1,1), scale = 0.1, size = size)
    second_quadrant = np.random.normal(loc = (-1,1), scale = 0.1, size = size)
    third_quadrant = np.random.normal(loc = (-1,-1), scale = 0.1, size = size)
    fourth_quadrant = np.random.normal(loc = (1,-1), scale = 0.1, size = size)
    data = np.concatenate([first_quadrant, second_quadrant, third_quadrant, fourth_quadrant], dtype = "float32")
    labels = np.concatenate([np.zeros(N), np.ones(N), np.ones(N)*2, np.ones(N)*3], dtype = "float32")
    return data, labels


def generate_data_double_cluster():
    size = (128, 2)
    N = size[0]
    first_quadrant = np.random.normal(loc = (1,1), scale = 0.1, size = size)
    second_quadrant = np.random.normal(loc = (-1,1), scale = 0.1, size = size)
    third_quadrant = np.random.normal(loc = (-1,-1), scale = 0.1, size = size)
    fourth_quadrant = np.random.normal(loc = (2, 2), scale = 0.1, size = size)
    data = np.concatenate([first_quadrant, second_quadrant, third_quadrant, fourth_quadrant], dtype = "float32")
    labels = np.concatenate([np.zeros(N), np.ones(N), np.ones(N)*2, np.ones(N)*3], dtype = "float32")
    return data, labels

def generate_random_data(n):
    size = (128, 2)
    N = size[0]
    quadrants = []
    for i in range(n):
        draw_theta = np.random.uniform(low = 0, high = 2*np.pi)
        x, y = np.cos(draw_theta), np.sin(draw_theta)
        quadrants.append(np.random.normal(loc = (x, y), scale = 0.1, size = size))
    data = np.concatenate(quadrants, dtype = "float32")
    labels = np.concatenate([np.ones(N) * i for i in range(n)], dtype = "float32")
    return data, labels

class UMIDataset(Dataset):
    def __init__(self, data, labels):
        #x, y = generate_data()
        self.data = data
        self.labels = labels
        #self.labels = F.one_hot(self.labels, num_classes=num_classes).to(float)
    def __len__(self): 
        return self.data.shape[0]
    def __getitem__(self, ix): 
        return self.data[ix], self.labels[ix]
