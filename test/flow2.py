from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.node import RemoteController
from mininet.cli import CLI

from topology import RandomTopology, FixedTopology
from random import random, randint
from time import sleep
import numpy as np


def initialize_topology(switch_num, p, k, r, sigma, t):
    topo = RandomTopology(low=switch_num, high=switch_num + 1, k=k, p=p, r=r, sigma=sigma, t=t)
    net = Mininet(topo, controller=lambda name: RemoteController(name, ip='127.0.0.1'))
    
    net.start()
    controller = net.get('c0')
    controller.cmdPrint('./enable_switches.sh ' + str(switch_num))
    print('Wait 5 seconds for enabling switches')
    sleep(5)

    return net, topo

def generate_flows(net, min_host_index, max_host_index, time):
    hpairs = []
    h_list = []
    H=0
    for i in range(min_host_index, max_host_index+1):
        src = "h"+str(i)
        src_host = net.get(src)
        src_host.cmd('iperf -s &')
        h_list.append(src_host)
        H += 1

    for src in h_list:
        for dst in h_list:
            if src == dst:
                continue
            src_host.cmd('iperf -c '+dst.IP()+' -t '+str(time) + ' &')
            print "Flow generated"
    sleep(time)

def main(switch_num_min, switch_num_max, p, k, r, sigma, t):
    net, topo = initialize_topology(switch_num_min, p, k, r, sigma, t)
    switch_num = len(topo.switches())
    host_num = len(topo.hosts())
    print('Just give the controller 30 seconds for topology discovery')
    sleep(30)
    print('Pingall to help the controller to fill flow tables')
    for i in range(50):
        print(str(i+1)+'/50')
        net.pingAll()
        sleep(2)
    # print('Wait 15 seconds to ensure flow tables are filled')
    # sleep(15)
    generate_flows(net, switch_num_max, switch_num_max + host_num - 1, 1000)


if __name__ == '__main__':
    setLogLevel('info')
    # simple_test(4, 0.5)
    main(20, 21, 0.3, 3, 80, 0.25, 10)