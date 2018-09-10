# Copyright (C) 2016 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import sys
from operator import attrgetter

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3

from ryu.lib import hub
from ryu.lib import stplib
from ryu.lib import dpid as dpid_lib
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet

from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link, get_host, get_all_host

from k_filter import InferenceModule

import numpy as np
import networkx as nx
import traceback

opts = {
    'p': '\033[95m',  # purple - HEADER
    'b': '\033[94m',  # blue - OKBLUE
    'B': '\033[44m',  # blue - OKBLUE
    'g': '\033[92m',  # green - OKGREEN
    'o': '\033[93m',  # orange - WARNING
    'r': '\033[91m',  # red - RED
    'd': '\033[1m',  # BOLD
    'u': '\033[4m'  # UNDERLINE
}


def printlog(text, parameters=None):
    endc = '\033[0m'
    if type(parameters) is str:
        for opt in opts:
            if opt in parameters:
                text = opts[opt] + text
    print(text + endc, end='')


class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'stplib': stplib.Stp}

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.switches = []
        self.switch_ports = []
        self.flows = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.n = 0
        self.number_responses = 0
        self.estimated_traffic = None
        # For shortest path routing
        self.mac_to_port = {}
        self.stp = kwargs['stplib']
        self.period = 1
        self.threshold = 75000
        self.alpha = 1
        self.ratio = 0.25
        self.history = []
        self.net = nx.DiGraph()
        self.clock = 0

    def _monitor(self):
        self.record = False
        self.manual_request = False
        with open('log.txt', 'r') as f:
            self.num_of_records = len(f.readlines())
        with open('alpha.txt', 'r') as f:
            num_lines_alpha = len(f.readlines())
        self.logger.info("sleep over!")
        index = 0

        while True:
            if not self.switches:
                hub.sleep(self.period)
                continue
            index += 1

            if index == 30:
                self.switch_link_map = self.get_switch_to_link_mapping()
            if index > 30:
                self.clock += 1
                self.query_all_switches()
                hub.sleep(self.period)
                index += 1

            hub.sleep(self.period)

            # if index == 40:
            #     print("It is time to initialize filter!")
            #     A = self.create_routing_matrix()
            #     print(A)
            # elif index < 40: # Note: REMOVE THIS CONDITION LATER!
            #     try:
            #         with open('log.txt', 'r') as f:
            #             current_num_of_records = len(f.readlines())
            #             if current_num_of_records != self.num_of_records:
            #                 self.record = not self.record
            #                 self.is_changed = True
            #             else:
            #                 self.is_changed = False
            #             self.num_of_records = current_num_of_records
            #         with open('alpha.txt', 'r') as f:
            #             current_num_lines_alpha = len(f.readlines())
            #             if current_num_lines_alpha > num_lines_alpha:
            #                 self.alpha = float(f.readlines()[-1])
            #                 self.logger.info('Alpha has changed to '+str(self.alpha))
            #                 num_lines_alpha = current_num_of_records
            #         self.select_switches(self.n, 0.8, 8, index)
            #     except:
            #         print(traceback.print_exc())
            # hub.sleep(self.period)

    def query_all_switches(self):
        for i in range(0, len(self.switches)):
            self._request_stats(i)

    def _request_stats(self, switch):
        datapath = self.switches[switch]
        if datapath is None:
            return
        # self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # req = parser.OFPFlowStatsRequest(datapath)
        # datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        switch_link_map = self.switch_link_map
        packet_count_file = open('topology1_t_'+str(self.clock), 'a')

        body = ev.msg.body
        switch = ev.msg.datapath.id
        for stat in sorted(body, key=attrgetter('port_no')):
            port = stat.port_no
            if switch in switch_link_map and port in switch_link_map[switch]:
                link_id = switch_link_map[switch][port]
                packet_count = stat.tx_packets
                packet_count_file.write("link %s: %s\n" % (link_id, packet_count))
                if link_id < len(self.links):
                    packet_count = stat.rx_packets
                    packet_count_file.write("link %s: %s\n" % (link_id+len(self.links), packet_count))
                else:
                    packet_count = stat.rx_packets
                    packet_count_file.write("link %s: %s\n" % (link_id-len(self.links), packet_count))

        # try:
        #     body = ev.msg.body
        #     switch = ev.msg.datapath.id
        #     if switch in self.selected_switches:
        #         for stat in sorted(body, key=attrgetter('port_no')):
        #             port = stat.port_no
        #             if port in self.switch_ports[switch]:
        #                 target_switch = self.switch_ports[switch][port]
        #                 # if self.last_measured[switch, target_switch] < self.query_count:
        #                 self.estimated_packet_count[switch, target_switch] = stat.tx_packets
        #                 self.last_measured[switch, target_switch] = self.query_count
        #                 # self.last_measured[switch, target_switch] = self.query_count
        #                 # if self.last_measured[target_switch, switch] < self.query_count:
        #                 #     self.estimated_packet_count[target_switch, switch] = stat.rx_packets
        #                 #     self.last_measured[target_switch, switch] = self.query_count

        #     for stat in sorted(body, key=attrgetter('port_no')):
        #         port = stat.port_no
        #         if port in self.switch_ports[switch]:
        #             target_switch = self.switch_ports[switch][port]
        #             self.real_packet_count[switch, target_switch] = stat.tx_packets
        #             if self.manual_request:
        #                 if switch not in self.link_packets:
        #                     self.link_packets[switch] = {}
        #                 self.link_packets[switch][port] = stat.tx_packets

        #     self.remaining_requests -= 1

        #     if self.remaining_requests == 0:
        #         self.estimate()
        # except:
        #     print(traceback.print_exc())

    def get_switch_to_link_mapping(self):
        """ returns a dictionary as follows:
            {switch1_id : {port1_no: link1_id, port2_no: link2_id, ...}, ...} """
        link_list = get_link(self, None)
        self.links = link_list
        switch_link_mapping = {}
        link_id = 0
        for link in link_list:
            if link.src.dpid not in switch_link_mapping:
                switch_link_mapping[link.src.dpid] = {}
            switch_link_mapping[link.src.dpid][link.src.port_no] = link_id
            if link.dst.dpid not in switch_link_mapping:
                switch_link_mapping[link.dst.dpid] = {}
            switch_link_mapping[link.dst.dpid][link.dst.port_no] = link_id + len(link_list)
            link_id += 1
        self.max_link_id = link_id
        return switch_link_mapping

    def get_link_packet_counts(self):
        """ Note: SHOULD Refactor the algorithm here
            returns a dictionary as follows:
            {link1_id: packet_count1, link2_id: packet_count2, ...} """
        link_list = get_link(self, None)
        link_packet_mapping = {}
        link_id = 0
        self.link_packets = {}
        self.manual_request = True
        for link in link_list:
            self._request_stats(link.src.dpid)
        hub.sleep(5) # wait for completion of asynchronous stats reply event

        switch_link_mapping = self.get_switch_to_link_mapping()
        for switch in switch_link_mapping:
            for port in switch_link_mapping[switch]:
                if switch in self.link_packets and port in self.link_packets[switch]:
                    link_packet_mapping[switch_link_mapping[switch][port]] = self.link_packets[switch][port]
        self.manual_request = False
        return link_packet_mapping

    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev=None):
        try:
            switch_list = get_switch(self, None)
            host_list = get_all_host(self)
            #print([h.mac for h in host_list])
            self.num_hosts = len(host_list)

            self.switches = [switch.dp for switch in switch_list]
            self.N = len(self.switches)
            self.num_nodes = self.N + self.num_hosts

            switches=[switch.dp.id for switch in switch_list]
            
            

            self.switches.insert(0, None)
            self.estimated_traffic = np.zeros((self.N + 1, self.N + 1))
            self.real_traffic = np.zeros((self.N + 1, self.N + 1))
            self.estimated_packet_count = np.zeros((self.N + 1, self.N + 1), dtype=np.int)
            self.real_packet_count = np.zeros((self.N + 1, self.N + 1), dtype=np.int)
            self.last_measured = np.zeros((self.N + 1, self.N + 1), dtype=np.int)
            n = int(self.ratio * self.N)
            n = 1 if n == 0 else n
            self.n = n

            link_list = get_link(self, None)
            self.link_list = link_list
            links=[(link.src.dpid,link.dst.dpid,{'port':link.src.port_no}) for link in link_list]

            self.net.add_nodes_from(switches)
            self.net.add_edges_from(links)

            self.switch_ports = {}
            self.switch_links = np.zeros((self.N + 1, self.N + 1), dtype=np.int)
            for link in link_list:
                if link.src.dpid in self.switch_ports:
                    self.switch_ports[link.src.dpid][link.src.port_no] = link.dst.dpid
                    self.switch_links[link.src.dpid, link.dst.dpid] = 1
                else:
                    self.switch_ports[link.src.dpid] = {link.src.port_no: link.dst.dpid}
                    self.switch_links[link.src.dpid, link.dst.dpid] = 1
                if link.dst.dpid in self.switch_ports:
                    self.switch_ports[link.dst.dpid][link.dst.port_no] = link.src.dpid
                    self.switch_links[link.dst.dpid, link.src.dpid] = 1
                else:
                    self.switch_ports[link.dst.dpid] = {link.dst.port_no: link.src.dpid}
                    self.switch_links[link.dst.dpid, link.src.dpid] = 1
        except:
            print(traceback.print_exc())

    def filter(self,o_list):
        if not self.E:
            self.E = 1
        if not self.V:
            self.V = 2

        n2 = self.num_nodes**2
        h2 = self.num_hosts**2
        v2 = self.V**2
        e2 = self.E**2

        self.create_routing_matrix()
        

        F = np.vstack((np.eye(h2),self.A))  # Transition matrix
        F = np.concatenate((F,np.zeros((n2+h2,n2))),axis=1)
        Q = np.eye(h2+n2)
        Q[:h2,:h2] = v2*Q[:h2,:h2]
        Q[h2:,h2:] = 0
        mq = np.zeros(n2+h2)
        
        H = np.concatenate((np.zeros((n2,h2)),np.eye(n2)),axis=1)
        R = e2*np.eye(n2)
        mr = np.zeros(n2)
        
        I = np.eye(n2+h2)
        
        x_list = []
            
        x = np.zeros(n2+h2)
        P = np.zeros((n2+h2,n2+h2))
        for o in o_list: # @ operator does now work in Python 2.x
            x = F.dot(x)
            P = ((F.dot(P)).dot(F.T)) + Q
            
            y = o - (H.dot(x))
            S =  R + ((H.dot(P)).dot(H.T))
            K = (P.dot(H.T)).dot(inv(S))  # Kalman gain
            x = x + (K.dot(y))
            P = (I - (K.dot(H))).dot(P)
            
            x_list.append(np.copy(x))
            
        return x_list

    def create_routing_matrix(self):
        switch_list = get_switch(self, None)
        host_list = get_all_host(self)

        nH = len(host_list)  # The number of hosts
        nS = len(switch_list)  # The number of switches
        N = nH + nS  # The number of nodes
        switches=[switch.dp.id for switch in switch_list]
        print(switches)
        print([host.dp.id for host in host_list])

        links=[str(link.src.dpid) + "->" + str(link.dst.dpid) for link in get_link(self, None)]

        print("host list: " + str(host_list))
        print("host switches: " + str([host.port.dpid for host in host_list]))
        print("nH: " + str(nH))
        print("nS: " + str(nS))
        print("links: " + str(links))


        switches = list(range(nS))  # The list of switch ids
        hosts = list(range(nS, N))  # The list of host ids

        self.net.add_nodes_from([h+1 for h in hosts])
        for i,host in enumerate(host_list):
            print("adding edge between " + str(host.port.dpid) + " and " + str(nS+i+1))
            self.net.add_edge(host.port.dpid,nS+i+1)

        edges = self.net.edges()
        nL = len(edges)
        L = 2*nL  # The number of directed links

        print("L: " + str(L))

        # Populate the dictionaries

        links = {(k-1): list(v.values()) for k,v in self.get_switch_to_link_mapping().items()}
        for i,host in enumerate(host_list):
            links[host.port.dpid-1].append(self.max_link_id)
            self.max_link_id += 1

        lpair2id = {}  # A mapping from edge pairs (src,dst) to link ids
        for i,edge in enumerate(edges):
            lpair2id[edge] = i
            lpair2id[(edge[1], edge[0])] = nL+i
            if edge[0] in links:
                links[edge[0]].append(i)
                links[edge[0]].append(nL+i)
            if edge[1] in links:
                links[edge[1]].append(i)
                links[edge[1]].append(nL+i)
                
        hpair2id = {}  # A mapping from host pairs (src,dst) to the corresponding flow ids
        i = 0
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    hpair2id[(src,dst)] = i
                    i += 1
        H2=i  # The number of flows

        print("Edges----")
        print(edges)
        # Create the routing matrix by using the shortest path algorithm
        A = np.zeros((L,H2))
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    try:
                        path = nx.shortest_path(self.net, src+1, dst+1)
                        for k in range(len(path)-1):
                            A[lpair2id[(path[k],path[k+1])],hpair2id[(src,dst)]] = 1
                    except:
                        continue
        self.A = A
        self.Inference = InferenceModule(A=A, links=links)

        return A

    def select_switches(self, n, threshold, period, query_count):
        t = query_count
        self.query_count = t
        switch_scores = np.zeros(self.N + 1)
        for switch in range(1, self.N + 1):
            # The maximum of incoming and outgoing packet rates
            switch_scores[switch] = np.max(self.estimated_traffic[switch, :])
            # switch_scores[switch] = max(max(self.estimated_traffic[switch, :]),
            #                                max(self.estimated_traffic[:, switch]))

        # Calculate the first normalization factor
        # z = np.sum(switch_scores)
        # z = 1. if z == 0 else z

        # Calculate the weights of the switches
        w = np.zeros(self.N + 1)
        for i in range(1, self.N + 1):
            w[i] = (1 - self.alpha) * (switch_scores[i] / (self.threshold / 2)) + self.alpha * (1. / self.N)
        w = w / sum(w)
        self.w = w
        # Choose a subset of the switches according to the weights
        selected_switches = np.random.choice(self.N + 1, n, False, w)

        self.remaining_requests = self.N
        self.prev_estimated_packet_count = np.copy(self.estimated_packet_count)
        self.prev_real_packet_count = np.copy(self.real_packet_count)
        self.prev_last_measured = np.copy(self.last_measured)
        self.selected_switches = selected_switches
        # for selected in selected_switches:
        for switch in range(1, self.N + 1):
            self._request_stats(switch)

    def estimate(self):
        # The code below using vector operations belonging to numpy
        time_vector = np.ones(self.N + 1) * self.query_count
        for switch in self.selected_switches:
            # Measure and update outgoing flows
            delta_time_src = time_vector - self.prev_last_measured[switch, :]  # [dt1 dt2 dt3 ...]
            delta_packet_src = self.estimated_packet_count[switch, :] - self.prev_estimated_packet_count[switch, :]
            for dest in [self.switch_ports[switch][port] for port in self.switch_ports[switch]]:
                self.estimated_traffic[switch, dest] = delta_packet_src[dest].astype(np.float) / (
                    delta_time_src[dest] * self.period)

        for switch in range(1, self.N + 1):
            delta_packet_real_src = self.real_packet_count[switch, :] - self.prev_real_packet_count[switch, :]
            for dest in [self.switch_ports[switch][port] for port in self.switch_ports[switch]]:
                self.real_traffic[switch, dest] = delta_packet_real_src[dest].astype(np.float) / self.period

        # np.set_printoptions(precision=3)
        # np.set_printoptions(suppress=True)
        # max_traffic = np.max(self.estimated_traffic)
        # if max_traffic > self.threshold:
        #     self.logger.info('Estimated Congestion:')
        #     self.logger.info(np.where(self.estimated_traffic == max_traffic))
        # max_traffic = np.max(self.real_traffic)
        # if max_traffic > self.threshold:
        colors = ['g', 'o', 'r']
        self.logger.info('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
        self.logger.info('==================================================')
        self.logger.info('==========       TRAFFIC MONITOR        ==========')
        self.logger.info('==================================================')
        self.logger.info('Switch scores:')
        self.logger.info('   ' + ', '.join(['%.2f' % w for w in self.w[1:]]))
        self.logger.info('Selected switches:')
        self.logger.info('   ' + ', '.join(map(str, sorted(self.selected_switches))))
        real_level = (2 * self.real_traffic / self.threshold).astype(np.int)
        real_level[real_level > 2] = 2
        estimated_level = (2 * self.estimated_traffic / self.threshold).astype(np.int)
        estimated_level[estimated_level > 2] = 2
        for i in range(1, self.real_traffic.shape[0]):
            background = 'B' if i in self.selected_switches else ''
            for j in range(1, self.real_traffic.shape[1]):
                if self.switch_links[i, j] == 1:
                    color_code = colors[real_level[i, j]] + background
                    traffic = ('%.1f' % self.real_traffic[i, j]).rjust(10)
                    printlog(traffic, color_code)
                else:
                    printlog('X'.rjust(10), 'b')
            # print('          ||||||||||          ',)
            # for j in range(1, self.real_traffic.shape[1]):
            #     if self.switch_links[i, j] == 1:
            #         color_code = colors[estimated_level[i, j]] + background
            #         traffic = ('%.1f' % self.estimated_traffic[i, j]).rjust(10)
            #         printlog(traffic, color_code)
            #     else:
            #         printlog('X'.rjust(10), 'b')
            print('')
        self.logger.info('\n\n\n\n\n\n\n\n\n\n')
        # self.logger.info(np.where(self.real_traffic == max_traffic))
        # self.logger.info('Estimated Traffic')
        # self.logger.info(self.estimated_traffic)
        # self.logger.info('Real Traffic')
        # self.logger.info(self.real_traffic)
        # self.logger.info('Real Packet Count')
        # self.logger.info(self.real_packet_count)
        if self.record:
            self.history.append((np.copy(self.real_traffic), np.copy(self.estimated_traffic)))
        elif self.is_changed:
            confusion_matrix = np.zeros((2, 2), dtype=np.int)
            for real_traffic, estimated_traffic in self.history:
                real_congestion = np.zeros((self.N + 1, self.N + 1), dtype=np.int)
                real_congestion[real_traffic > self.threshold] = 1
                estimated_congestion = np.zeros((self.N + 1, self.N + 1), dtype=np.int)
                estimated_congestion[estimated_traffic > self.threshold] = 1
                for i in range(1, self.real_traffic.shape[0]):
                    for j in range(1, self.real_traffic.shape[1]):
                        if self.switch_links[i, j] == 1:
                            confusion_matrix[estimated_congestion[i, j], real_congestion[i, j]] += 1
            with open('confusion_matrix.txt', 'w') as h:
                h.write(str(confusion_matrix))
            self.logger.info('Confusion matrix has been created!')

    def delete_flow(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        for dst in self.mac_to_port[datapath.id].keys():
            match = parser.OFPMatch(eth_dst=dst)
            mod = parser.OFPFlowMod(
                datapath, command=ofproto.OFPFC_DELETE,
                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                priority=1, match=match)
            datapath.send_msg(mod)

    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(stplib.EventTopologyChange, MAIN_DISPATCHER)
    def _topology_change_handler(self, ev):
        self.get_topology_data()

    @set_ev_cls(stplib.EventPortStateChange, MAIN_DISPATCHER)
    def _port_state_change_handler(self, ev):
        self.get_topology_data()
