#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import networkx as nx

from itertools import chain


def softmax(activations):
    """Normalize floats with softmax transformation.
    Parameters
    ----------
    activations array or list of floats
    """
    activations = [np.exp(activation) for activation in activations]
    activations = [float(activation/sum(activations)) for activation in activations]
    return activations


def plot_graph(weights, display = True, output_file = None, output_format = "pdf"):
    """Plot a two dimensional array of weights.
    Parameters
    ----------
    display:        boolean, optional
                    show the plot of the graph
    output_file:    str or path, optional
                    path to file to save graph in, does not save per default
    output_format:  "pdf", "png" or "jpg"
    """
    cues = weights.index
    outcomes = weights.columns

    minimum = np.min(weights.values)
    maximum = np.max(weights.values)

    # colour code edge weights
    normalization = colors.Normalize(vmin = minimum, vmax = maximum)
    scale = cm.ScalarMappable(norm = normalization, cmap = cm.seismic)
    colours = scale.to_rgba(weights)
    colours = np.ndarray.flatten(colours).reshape(weights.size,-1)

    # structure graph
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(cues, bipartite = 0)
    bipartite_graph.add_nodes_from(outcomes, bipartite = 1)
    bipartite_graph.add_edges_from([(cue, outcome) for cue in cues for outcome in outcomes])
    left, right = nx.bipartite.sets(bipartite_graph)

    # make positions for each node
    # (multiply by len() of other partition to center)
    positions = dict()
    for idx, node in enumerate(left):
        positions[node] = (1, len(outcomes)*idx)
    for idx, node in enumerate(right):
        positions[node] = (2, len(cues)*idx)

    # make dictionary with labels
    labels = dict()
    for idx, cue in enumerate(chain(cues, outcomes)):
        labels[cue] = cue

    # shade nodes depending on activation level
    # (cues have fixed activation)
    activations = weights.sum(0)
    activations_normalized = softmax(activations)
    alphas = [0.5]*len(cues) + activations_normalized
    gray_values = cm.gray_r(alphas)

    # draw
    nx.draw(bipartite_graph, pos = positions, edge_color = colours, labels = labels, node_size = 400, node_shape = "o", node_color = gray_values)

    # display and save
    if display:
        plt.show()
    if output_file:
        plt.savefig(output_file, format = output_format)
