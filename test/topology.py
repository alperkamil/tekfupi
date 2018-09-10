from mininet.topo import Topo
# from topology import get_topology
import numpy as np
import networkx as nx

import networkx as nx

def get_topology():
    nH = 10  # The number of hosts
    nS = 10  # The number of switches
    N = nH + nS  # The number of nodes

    # Create a graph of switches
    G = nx.star_graph(nS-1)
    # G = nx.complete_graph(nS)

    switches = list(range(nS))  # The list of switch ids
    hosts = list(range(nS, N))  # The list of host ids

    # Append hosts
    for i in range(nH):
        host = nS + i
        G.add_node(host)
        G.add_edge(i, host)

    return G, nH, nS
    

# class RandomTopology( Topo ):
#     "Our random topology to be used in the Mininet Simulation"

#     def __init__( self, low=16, high=17, k=3, p=0.3, r=8, sigma=0.25, t=1024, threshold=128,
#                  alpha_values=np.arange(0.1, 1.1, 0.1) ):
#         "Create random topology."

#         # Initialize topology
#         Topo.__init__( self )

#         self.low = low
#         self.high = high
#         self.k = k                          # Not used for now
#         self.p = p
#         self.r = r                          # Not used for now
#         self.sigma = sigma                  # Not used for now
#         self.T = t                          # Not used for now
#         self.threshold = threshold          # Not used for now
#         self.alpha_values = alpha_values    # Not used for now

#         # Generate a random network
#         self.num_switches = np.random.randint(self.low, self.high)
#         self.num_hosts = np.random.poisson(self.num_switches)
#         self.num_nodes = self.num_switches + self.num_hosts


#         # self.G = nx.connected_watts_strogatz_graph(self.num_switches, self.k, self.p)
#         self.G = nx.balanced_tree(self.num_switches,int(np.log2(self.num_switches)))

#         # Append hosts
#         host_switches = np.random.choice(self.num_switches, self.num_hosts)
#         for i in range(self.num_hosts):
#             host = self.num_switches + i
#             self.G.add_node(host)
#             self.G.add_edge(host_switches[i], host)

#         #nx.draw(self.G)
#         #plt.show()

#         # Add switches
#         self.switch_list = []
#         for i in range(1, self.num_switches+1):
#             self.switch_list.append(self.addSwitch('s'+str(i)))

#         # Add hosts
#         self.host_list = []
#         for i in range(self.num_switches+1, self.num_nodes+1):
#             self.host_list.append(self.addHost('h'+str(i)))

#         for edge in self.G.edges():
#             if edge[0] < self.num_switches and edge[1] < self.num_switches:
#                 self.addLink(self.switch_list[edge[0]], self.switch_list[edge[1]])
#             elif edge[0] >= self.num_switches and edge[1] < self.num_switches:
#                 self.addLink(self.host_list[edge[0]-self.num_switches], self.switch_list[edge[1]])
#             elif edge[0] < self.num_switches and edge[1] >= self.num_switches:
#                 self.addLink(self.switch_list[edge[0]], self.host_list[edge[1]-self.num_switches])



        # # Connect hosts to switches
        # host_switches = np.random.choice(self.switch_list, self.num_hosts)
        # for i in range(0, self.num_hosts):
        #     self.addLink(self.host_list[i], host_switches[i])

        # # Connect switches
        # for i in range(0, self.num_switches):
        #     for j in range(0, i):
        #         if np.random.random() < self.p:
        #             self.addLink(self.switch_list[i], self.switch_list[j])

class RandomTopology( Topo ):
    "Our random topology to be used in the Mininet Simulation"

    def __init__( self, low=16, high=17, k=3, p=0.3, r=8, sigma=0.25, t=1024, threshold=128,
                 alpha_values=np.arange(0.1, 1.1, 0.1) ):
        "Create random topology."

        # Initialize topology
        Topo.__init__( self )

        self.low = low
        self.high = high
        self.k = k                          # Not used for now
        self.p = p
        self.r = r                          # Not used for now
        self.sigma = sigma                  # Not used for now
        self.T = t                          # Not used for now
        self.threshold = threshold          # Not used for now
        self.alpha_values = alpha_values    # Not used for now

        # Generate a random network
        self.num_switches = np.random.randint(self.low, self.high)
        self.num_hosts = np.random.poisson(self.num_switches)
        self.num_nodes = self.num_switches + self.num_hosts


        self.G = nx.connected_watts_strogatz_graph(self.num_switches, self.k, self.p)

        # Append hosts
        host_switches = np.random.choice(self.num_switches, self.num_hosts)
        for i in range(self.num_hosts):
            host = self.num_switches + i
            self.G.add_node(host)
            self.G.add_edge(host_switches[i], host)

        #nx.draw(self.G)
        #plt.show()

        # Add switches
        self.switch_list = []
        for i in range(1, self.num_switches+1):
            self.switch_list.append(self.addSwitch('s'+str(i)))

        # Add hosts
        self.host_list = []
        for i in range(self.num_switches+1, self.num_nodes+1):
            self.host_list.append(self.addHost('h'+str(i)))

        for edge in self.G.edges():
            if edge[0] < self.num_switches and edge[1] < self.num_switches:
                self.addLink(self.switch_list[edge[0]], self.switch_list[edge[1]])
            elif edge[0] >= self.num_switches and edge[1] < self.num_switches:
                self.addLink(self.host_list[edge[0]-self.num_switches], self.switch_list[edge[1]])
            elif edge[0] < self.num_switches and edge[1] >= self.num_switches:
                self.addLink(self.switch_list[edge[0]], self.host_list[edge[1]-self.num_switches])



        # # Connect hosts to switches
        # host_switches = np.random.choice(self.switch_list, self.num_hosts)
        # for i in range(0, self.num_hosts):
        #     self.addLink(self.host_list[i], host_switches[i])

        # # Connect switches
        # for i in range(0, self.num_switches):
        #     for j in range(0, i):
        #         if np.random.random() < self.p:
        #             self.addLink(s

class FixedTopology( Topo ):
    "Our fixed topology to be used in the Mininet Simulation"

    def __init__(self):
        "Create fixed topology."

        # Initialize topology
        Topo.__init__( self )

        # Generate a network
        self.G, self.num_hosts, self.num_switches = get_topology()
        self.num_nodes = self.num_hosts + self.num_switches

        # Add switches
        self.switch_list = []
        for i in range(1, self.num_switches+1):
            self.switch_list.append(self.addSwitch('s'+str(i)))

        # Add hosts
        self.host_list = []
        for i in range(self.num_switches+1, self.num_nodes+1):
            self.host_list.append(self.addHost('h'+str(i)))

        for edge in self.G.edges():
            if edge[0] < self.num_switches and edge[1] < self.num_switches:
                self.addLink(self.switch_list[edge[0]], self.switch_list[edge[1]])
            elif edge[0] >= self.num_switches and edge[1] < self.num_switches:
                self.addLink(self.host_list[edge[0]-self.num_switches], self.switch_list[edge[1]])
            elif edge[0] < self.num_switches and edge[1] >= self.num_switches:
                self.addLink(self.switch_list[edge[0]], self.host_list[edge[1]-self.num_switches])

topos = { 'randomtopo': ( lambda: RandomTopology() ), 'fixedtopo': ( lambda: FixedTopology() )  }
