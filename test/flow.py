from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.node import RemoteController
from mininet.cli import CLI

from topology import RandomTopology, FixedTopology
from random import random, randint
from time import sleep
import numpy as np

def generate_flows_alternative(net, min_host_index, max_host_index):
    hpairs = []
    for i in range(min_host_index, max_host_index+1):
        src = "h"+str(i)
        src_host = net.get(src)
        for j in range(min_host_index, max_host_index+1):
            src_host.cmd('flowgrindd -p ' + str(5999 + j)) # start flowgrind daemon
            if i != j:
                dst = "h"+str(j)
                hpairs.append(str(src)+'-'+str(dst))
    pair_num = len(hpairs)

    flow_num = int((pair_num/50) + (pair_num/20)*np.random.rand())

    ind = 0
    while True:
        ind += 1
        print "Round " + str(ind) + " starts"
        flow_list = np.random.choice(hpairs, len(hpairs), replace=False)
        for flow_c in flow_list:
            flow = flow_c.split('-')
            src_host, dst_host = net.get(flow[0], flow[1])
            src_ind, dst_ind = int(flow[0][1:]), int(flow[1][1:])
            rate = np.random.poisson(30) # gaussian distribution for packet rate, kBps
            time = np.random.poisson(30) # poisson for transmission time, seconds
            src_host.cmd('flowgrind -H s=%s/%s:%s,d=%s/%s:%s -R s=%s -T s=%s' % (src_host.IP(), src_host.IP(), dst_ind, dst_host.IP(), dst_host.IP(), src_ind, str(rate)+'kB', time))
            print "Flow generated - " + str(time) + " seconds"
        print "Round " + str(ind) + " starts"


def initialize_topology(switch_num, p, k, r, sigma, t):
    topo = RandomTopology(low=switch_num, high=switch_num + 1, k=k, p=p, r=r, sigma=sigma, t=t)
    net = Mininet(topo, controller=lambda name: RemoteController(name, ip='127.0.0.1'))
    
    net.start()
    controller = net.get('c0')
    controller.cmdPrint('./enable_switches.sh ' + str(switch_num))
    print('Wait 5 seconds for enabling switches')
    sleep(5)

    return net, topo

def generate_flows(net, min_host_index, max_host_index):
    hpairs = []
    for i in range(min_host_index, max_host_index+1):
        src = "h"+str(i)
        src_host = net.get(src)
        src_host.cmd('flowgrindd') # start flowgrind daemon
        for j in range(min_host_index, max_host_index+1):
            if i != j:
                dst = "h"+str(j)
                hpairs.append(str(src)+'-'+str(dst))
    pair_num = len(hpairs)

    flow_num = int((pair_num/50) + (pair_num/20)*np.random.rand())

    ind = 0
    while True:
        ind += 1
        print "Round " + str(ind) + " starts"
        flow_list = np.random.choice(hpairs, len(hpairs), replace=False) # using all of the pairs might be overkill
        flowgrind_arg = ""
        for flow_c in flow_list:
            flow = flow_c.split('-')
            src_host, dst_host = net.get(flow[0], flow[1])
            rate = np.random.poisson(30) # gaussian distribution for packet rate, kBps
            time = np.random.poisson(20) # poisson for transmission time, seconds
            flowgrind_arg += " -H s=%s,d=%s -R s=%s -T s=%s" % (src_host.IP(), dst_host.IP(), str(rate)+"kB", time) 

        controller = net.get('c0')
        controller.cmdPrint('flowgrind' + flowgrind_arg)
        print "Round " + str(ind) + " ends"

    # while True:
    #     flow_list = np.random.choice(hpairs, len(hpairs), replace=False)
    #     for flow_c in flow_list:
    #         flow = flow_c.split('-')
    #         src_host, dst_host = net.get(flow[0], flow[1])
    #         rate = np.random.poisson(30) # gaussian distribution for packet rate, Mbps
    #         time = np.random.poisson(30) # poisson for transmission time, seconds
    #         src_host.cmd('flowgrind -H s='+src_host.IP() + ',d='+dst_host.IP() + ' -G s=g:E:' + str(rate) + ' -T s=' + str(time) + ' &')
    #         print "Flow generated - " + str(time) + " seconds"
    #     print 'going to sleep for 30 seconds'
    #     sleep(30)

def main(switch_num_min, switch_num_max, p, k, r, sigma, t):
    net, topo = initialize_topology(switch_num_min, p, k, r, sigma, t)
    print('Just give the controller 30 seconds for topology discovery')
    sleep(30)
    print('Pingall to help the controller to fill flow tables')
    net.pingAll()
    print('Wait 30 seconds to ensure flow tables are filled')
    sleep(30)
    switch_num = len(topo.switches())
    host_num = len(topo.hosts())
    generate_flows(net, switch_num_max, switch_num_max + host_num - 1)
    #generate_flows_alternative(net, switch_num_max, switch_num_max + host_num - 1)


if __name__ == '__main__':
    setLogLevel('info')
    # simple_test(4, 0.5)
    main(6, 7, 0.3, 3, 80, 0.25, 10) # change this later