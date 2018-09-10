#!/usr/bin/env python
# coding=utf-8
"""Provides the simulation environment depending random walk process flows and weighted selection of switches.

__author__ = "Alper Kamil Bozkurt, Gökcan Çantalı"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alper Kamil Bozkurt"
__email__ = "alperkamil.bozkurt@gmail.com

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import copy
import os


class Simulation:
    def __init__(self, low=16, high=33, k=3, p=0.5, r=8, sigma=0.25, t=1024, threshold=128,
                 alpha_values=np.arange(0.1, 1.1, 0.1)):
        """Generates a random network according to given parameters.
        :param low: The smallest number of switches (inclusive).
        :param high: The largest number of switches (exclusive).
        :param k: Each node is joined with its ``k`` nearest neighbors in a ring topology.
        :param p: The probability of rewiring each edge.
        :param r: The ratio of the total number of switches to the number of switches to be measured.
        :param sigma: # The standard deviation of the Gaussian random variable.
            to be added to packet rates in each time step.
        :param t: The number of time steps to be used in simulation.
        :param threshold: # The congestion threshold.
        :param alpha_values: The list of alpha values.
        """
        self.low = low
        self.high = high
        self.k = k
        self.p = p
        self.r = r
        self.sigma = sigma
        self.T = t
        self.threshold = threshold
        self.alpha_values = alpha_values

        # Fix the seed for debugging
        # # np.random.seed(123457)

        # Generate a network
        self.num_switches = np.random.randint(self.low, self.high)
        self.num_hosts = np.random.poisson(self.num_switches)
        self.num_nodes = self.num_switches + self.num_hosts
        self.G = nx.connected_watts_strogatz_graph(self.num_switches, self.k, self.p)

        self.switches = range(self.num_switches)  # The list of switch ids
        self.hosts = range(self.num_switches, self.num_nodes)  # The list of host ids

        # Append hosts
        host_switches = np.random.choice(self.num_switches, self.num_hosts)
        for i in range(self.num_hosts):
            host = self.num_switches + i
            self.G.add_node(host)
            self.G.add_edge(host_switches[i], host)

        # The number of switches to be measured in each time step
        self.n = self.num_switches / self.r

        # Draw the graph
        # nx.draw(self.G)
        # plt.show()
        self.draw()

        print 'Number of Switches:', self.num_switches
        print 'Number of hosts:', self.num_hosts
        print 'Number of links:', len(self.G.edges()) + self.num_hosts
        print '=========='

        # Create flow paths
        self.flows = {}
        self.paths = {}
        for src in self.hosts:
            for dest in self.hosts:
                if src == dest:
                    continue
                self.paths[(src, dest)] = nx.shortest_path(self.G, src, dest)
                self.flows[(src, dest)] = 0.

    def draw(self):
        pos = nx.random_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.switches,
                               node_color='r', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.hosts,
                               node_color='g', node_size=500, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, width=2.0, alpha=0.5)
        plt.show()

    def simulate(self, alpha=None):
        """Runs experiments for different alpha values
        :return:
        """
        if alpha:
            alpha_values = [alpha]
        else:
            alpha_values = self.alpha_values

        for alpha in alpha_values:
            # Initialize the network state
            real_packet_count = np.array([[0] * self.num_nodes] * self.num_nodes)
            for pair in self.flows:
                self.flows[pair] = 0

            # Initialize the matrices to be used in estimation
            estimated_traffic = np.array([[0.] * self.num_nodes] * self.num_nodes)
            estimated_packet_count = np.array([[0] * self.num_nodes] * self.num_nodes)
            last_measured = np.array([[-1] * self.num_nodes] * self.num_nodes)

            # Initialize the evaluation variables
            congestion_time = {}
            congestion_count = 0
            response_time = 0.

            # Start the simulation
            for t in range(self.T):
                # Create a clear traffic matrix
                real_traffic = np.array([[0.] * self.num_nodes] * self.num_nodes)
                # Calculate the packet rate on the links and update packet counts
                for pair in self.flows:
                    rate = self.flows[pair]
                    self.flows[pair] = max(0, rate + self.sigma * np.random.randn())  # Update the future rate
                    path = self.paths[pair]
                    # Update all the links on the path
                    for i in range(len(path) - 1):
                        real_packet_count[path[i]][path[i + 1]] += int(rate)
                        real_traffic[path[i], path[i + 1]] += int(rate)

                # Calculate the switch scores
                switch_scores = []
                for switch in self.switches:
                    switch_scores.append(max(max(estimated_traffic[switch, :]), max(estimated_traffic[:, switch])))

                # Calculate the first normalization factor
                z = np.sum(switch_scores)
                z = 1. if z == 0 else z

                # Calculate the weights of thw switches
                w = np.array([0.] * len(self.switches))
                for i in range(len(self.switches)):
                    w[i] = (1 - alpha) * (switch_scores[i] / (self.threshold / 2)) + alpha * (1. / len(self.switches))
                w = w / sum(w)

                # Choose a subset of the switches according to the weights
                selected_switches = np.random.choice(self.switches, self.n, False, w)
                time_vector = np.array([t] * self.num_nodes)  # Current time vector: [t t t ...]
                # The code below using vector operations belonging to numpy
                for switch in selected_switches:
                    # Measure and update incoming flows
                    delta_time_src = time_vector - last_measured[switch, :]  # [dt1 dt2 dt3 ...]
                    delta_packet_src = real_packet_count[switch, :] - estimated_packet_count[switch, :]
                    for dest in self.switches:
                        if delta_time_src[dest] == 0:
                            # Which means (switch,dest) link was updated by another observed switch.
                            continue
                        estimated_traffic[switch, dest] = delta_packet_src[dest] / delta_time_src[dest]
                    estimated_packet_count[switch, :] = real_packet_count[switch, :]
                    last_measured[switch, :] = time_vector

                    # Measure and update outgoing flows
                    delta_time_dest = time_vector - last_measured[:, switch]  # [dt1 dt2 dt3 ...]
                    delta_packet_dest = real_packet_count[:, switch] - estimated_packet_count[:, switch]
                    for src in self.switches:
                        if delta_packet_dest[src] == 0:
                            # Which means (src,switch) link was updated by another observed switch.
                            continue
                        estimated_traffic[src, switch] = delta_packet_dest[src] / delta_time_dest[src]
                    estimated_packet_count[:, switch] = real_packet_count[:, switch]

                # Detect new congested links
                real_congestion = np.nonzero(real_traffic >= self.threshold)
                num_congestion = len(real_congestion[0])
                for i in range(num_congestion):
                    link = real_congestion[0][i], real_congestion[1][i]
                    if link not in congestion_time:
                        congestion_time[link] = t
                old_congestion_links = []
                for link in congestion_time:
                    if real_traffic[link[0]][link[1]] < self.threshold:
                        old_congestion_links.append(link)
                for link in old_congestion_links:
                    del congestion_time[link]

                # Estimate new congested links
                estimated_congestion = np.nonzero(estimated_traffic >= self.threshold)
                num_congestion = len(estimated_congestion[0])
                for i in range(num_congestion):
                    link = estimated_congestion[0][i], estimated_congestion[1][i]
                    if link not in congestion_time:  # For now, ignore the false positives.
                        # print 'False positive, at time step: ', t
                        # print 'Link: ', pair, ' Estimated Rate: ', estimated_traffic[pair],
                        # ' Actual Rate: ', real_traffic[pair]
                        continue
                    else:
                        congestion_count += 1
                        response_time += (t - congestion_time[link])
                        del congestion_time[link]

                        # Congestion handling
                        src, dest = link

                        responsible_pairs = [pair for pair in self.paths
                                             if src in self.paths[pair] and dest in self.paths[pair] and
                                             self.paths[pair].index(dest) - self.paths[pair].index(src) == 1]
                        max_flow = 0
                        for pair in responsible_pairs:
                            if self.flows[pair] > max_flow:
                                max_flow = self.flows[pair]
                                # most_responsible_pair = pair
                            self.flows[pair] = 0

                            # flows[most_responsible_pair] = 0

            # filename = 'results'+os.sep+'r='+str(self.r)+'_threshold='+str(self.threshold)+'_sigma='+str(self.sigma)+'_alpha='+str(alpha)+'.txt'
            # output = open(filename, 'w')
            # output.write('Alpha: ' + str(alpha))
            # output.write('\nThe number of congestions : ' + str(congestion_count))
            # output.write('\nThe average response time : ' + str(response_time / max(1, congestion_count)))
            # output.close()

            print 'Alpha: ', alpha
            print 'The number of congestions : ', congestion_count
            print 'The average response time : ', response_time / max(1, congestion_count)
            print '=========='
        return response_time / max(1, congestion_count)


def simulate(s, alpha):
    new_s = copy.deepcopy(s)
    return s.simulate(alpha)


def simulate_multiple_params():
        s = Simulation()
        r_values = [2, 4, 8, 16]
        threshold_values = [128, 64, 256, 100, 150]
        sigma_values = [0.25, 0.50, 0.75, 1.00]
        alpha_values = np.arange(0.1, 1.1, 0.1)
        for r in r_values:
            s.r = r
            for threshold in threshold_values:
                s.threshold = threshold
                for sigma in sigma_values:
                    s.sigma = sigma
                    latency_list = Parallel(n_jobs=4, backend='threading')(
                        delayed(simulate)(s, alpha) for alpha in alpha_values)
                    print latency_list
                    with open('python_simulation_results.txt', 'a') as f:
                        f.write('r: ' + str(r) + ',  threshold: ' + str(threshold) + ',  sigma: ' + str(sigma) + '\n')
                        f.write(str(latency_list))
                        f.write('\n')


if __name__ == "__main__":
    simulate_multiple_params()
