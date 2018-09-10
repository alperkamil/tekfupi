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