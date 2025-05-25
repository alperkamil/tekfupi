# Network State Estimation for Congestion Detection in SDN
This repository provides an implementation of our approach described [here](https://dl.acm.org/doi/abs/10.1145/3154970.3154977) and a simulation environment for the experiments.

In this project, we use estimation theory or machine learning to estimate network state in SDN. This information will be utilized by a congestion detection scheme which will try to locate where the congestion is. The work will be tested in a simulation environment. If time allows, we will also try to present an analytical study for estimation and/or congestion performance.

In order to use our random topology class in mininet simulation, go into
the directory where random_topology.py is placed and run the following:

sudo mn --custom random_topology.py --topo randomtopo --controller remote

If you want to run the code without changing directory:

sudo mn --custom <path_to_random_topology.py> --topo randomtopo --controller remote

To run the conroller:

ryu-manager advanced_monitor_13.py 