from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.node import RemoteController
from mininet.cli import CLI

from random_topology import RandomTopology, FixedTopology
from random import random, randint
from time import sleep


def simple_test(switch_num, prob):
    topo = RandomTopology(low=switch_num, high=switch_num + 1, p=prob)
    net = Mininet(topo)
    # dumpNodeConnections(net.hosts)

    # h17, h20 = net.get('h17', 'h20')  
    # h1.cmd('iperf3 -c ' + h2.IP() + ' -u -n 10M -b 1M')


def generate_flows(net, min_host_index, max_host_index, flow_number, weibull_param1=None, weibull_param2=None):
    src = 'h' + str(randint(min_host_index, max_host_index))
    dst = 'h' + str(randint(min_host_index, max_host_index))
    while src == dst:
        dst = 'h' + str(randint(min_host_index, max_host_index))

    with open('mininet.txt', 'r') as m:
        previous_number_of_lines = len(m.readlines())
    while True:
        sleep(2)
        with open('mininet.txt', 'r') as m:
            lines = m.readlines()
            number_of_lines = len(lines)
            if number_of_lines == previous_number_of_lines:
                continue
            previous_number_of_lines = number_of_lines

            flow_number = -1
            if number_of_lines > 0:
                try:
                    flow_number = int(lines[-1])
                except:
                    pass
                if flow_number == 0:
                    break

        if flow_number == -1:
            continue

        src_host, dst_host = net.get(src, dst)
        with open('log.txt', 'a') as l:
            l.write('Flow starts...\n')
        print('The flow will start in 5 seconds...')
        sleep(5)  # sleep 10 seconds before flow generation
        src_host.cmd('flowgrindd')
        dst_host.cmd('flowgrindd')
        print('The flow has started.')
        for i in range(0, flow_number):  # randomly generate flows
            print('Flow: ' + str(i + 1))
            if not weibull_param1 or not weibull_param2:
                output_flow = src_host.cmd('flowgrind -H s='+src_host.IP() + ',d='+dst_host.IP())
            else:
                output_flow = src_host.cmd('flowgrind -H s='+src_host.IP() + ',d='+dst_host.IP() + ' -G s=q:W:'+str(weibull_param1)+':'+str(weibull_param2))
            print('Flow: ' + str(i + 1))
            output_flow = src_host.cmd('flowgrind -H s=' + src_host.IP() + ',d=' + dst_host.IP())
            # print('flow generation done! ')
            # print(output_flow)
            # output_dst = dst_host.cmd('iperf3 -s')
            # output_src = src_host.cmd('iperf3 -c ' + dst_host.IP() + ' -u -b 1M -k 1K &')
            # net.iperf((src_host, dst_host))

        print('The flow will finish in 15 seconds...')
        sleep(15)  # sleep 10 seconds after flow generation
        with open('log.txt', 'a') as l:
            l.write('Flow ended...\n')
        print('The flow has finished.')

            # print('output_src: ' + str(output_src))
            # print('output_dst: ' + str(output_dst))


def init_tests(switch_num_min, switch_num_max, p, k, r, sigma, t, threshold):
    #topo = RandomTopology(low=switch_num_min, high=switch_num_max, k=k, p=p, r=r, sigma=sigma, t=t, threshold=threshold)
    topo = FixedTopology()
    net = Mininet(topo=topo, controller=lambda name: RemoteController(name, ip='127.0.0.1'))
    net.start()
    controller = net.get('c0')
    controller.cmdPrint('./enable_switches.sh ' + str(switch_num_min))
    sleep(30)

    switch_num = len(topo.switches())
    host_num = len(topo.hosts())

    print('starting!')
    print('pingall')
    net.pingAll()
    net.pingAll()
    net.pingAll()
    net.pingAll()
    net.pingAll()
    #generate_flows(net, switch_num + 1, switch_num + host_num, 10)
    generate_flows(net, switch_num + 1, switch_num + host_num, 10, 1, 3) # with Weibull
    CLI(net)


if __name__ == '__main__':
    setLogLevel('info')
    # simple_test(4, 0.5)
    init_tests(10, 10, 0.3, 3, 8, 0.25, 1024, 128)
