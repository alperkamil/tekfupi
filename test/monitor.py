import simple_monitor_stp_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.topology.switches import Link
from ryu.topology.api import get_switch, get_link, get_host, get_all_host
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib import stplib
from operator import attrgetter

import numpy as np
from kalman_filter import InferenceModule
import pickle
from datetime import datetime


class Monitor(simple_monitor_stp_13.SimpleMonitor13):

    def __init__(self, *args, **kwargs):
        super(Monitor, self).__init__(*args, **kwargs)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        super(Monitor, self)._port_stats_reply_handler(ev)

    def _monitor(self):
        hub.sleep(200)  # Wait for the topology discovery

        host_list = get_host(self, None)
        switch_list = get_switch(self, None)
        link_list = get_link(self, None).keys()

        # Add missing backward links
        link_pairs = set([(link.src.dpid,link.dst.dpid) for link in link_list])
        missing_links = [Link(link.dst,link.src) for link in link_list if not (link.dst.dpid,link.src.dpid) in link_pairs]
        link_list.extend(missing_links)

        H, S = (len(host_list), len(switch_list))

        # The number of nodes, the number of flows and the number of links respectively
        # +2*H is for the links connecting switches and hosts (bidrectional)
        N, H2, L = (H + S, H * (H - 1), len(link_list) + 2 * H) 

        # Gather Flow Table Rules
        self.port_dst_to_rule = {}
        self.R, self.t = (0,None)
        for dp in self.datapaths.values():
            self._request_stats(dp)
        hub.sleep(5)

        # self.port_to_link[switch_id][port_no] gives the corresponding link id
        # of the port of the switch
        self.port_to_link = {}

        # self.port_to_host[switch_id][port_no] gives the corresponding host of
        # the port of the switch
        self.port_to_host = {}

        # switch_to_links[switch_id] provides the id list of the links
        # connected to the switch
        switch_to_links = {}

        # switch_to_rules[switch_id] provides the id list of the rules
        # in the switch
        switch_to_rules = {}

        # Populate the dictionaries
        for link_id, link in enumerate(link_list):
            self.port_to_link.setdefault(link.src.dpid, {})
            self.port_to_link[link.src.dpid][link.src.port_no] = link_id, link

            switch_to_links.setdefault(link.src.dpid, [])
            switch_to_links[link.src.dpid].append(link_id)
            switch_to_links.setdefault(link.dst.dpid, [])
            switch_to_links[link.dst.dpid].append(link_id)

        # Add hosts with the links
        for host_id, host in enumerate(host_list):
            rlink_id = len(link_list) + host_id  # (Host -> Switch)
            tlink_id = rlink_id + H  # (Switch -> Host)
            switch_to_links[host.port.dpid].extend([rlink_id, tlink_id])

            self.port_to_host.setdefault(host.port.dpid, {})
            self.port_to_host[host.port.dpid][host.port.port_no] = rlink_id, tlink_id, host

        for switch_id in self.port_dst_to_rule:
            switch_to_rules[switch_id] = [rule_id for rule_id, port_no in self.port_dst_to_rule[switch_id].values()]

        # Create the routing matricess
        A = np.zeros((L, H2))
        B = np.zeros((self.R, H2))
        flow_id = 0
        for src_id, src in enumerate(host_list):
            for dst_id, dst in enumerate(host_list):
                if src_id != dst_id:

                    # Outgoing link id of src host (Host -> Switch)
                    link_id = len(link_list) + src_id
                    A[link_id, flow_id] = 1

                    # The first switch on the path
                    switch_id = src.port.dpid  

                    # The corresponding rule and out port
                    rule_id, port_no = self.port_dst_to_rule[switch_id][src.port.port_no,dst.mac]
                    B[rule_id,flow_id] = 1

                    # Populate through the switches on the path
                    while port_no in self.port_to_link[switch_id]:

                        # Next Link
                        link_id, link = self.port_to_link[switch_id][port_no]
                        A[link_id, flow_id] = 1

                        # Next Switch
                        switch_id = link.dst.dpid

                        rule_id, port_no = self.port_dst_to_rule[switch_id][link.dst.port_no,dst.mac]
                        B[rule_id,flow_id] = 1

                    # Incoming link id of the dst host (Switch -> Host)
                    link_id = len(link_list) + H + dst_id
                    A[link_id, flow_id] = 1

                    flow_id += 1

        im = InferenceModule(A, links=switch_to_links)
        im2 = InferenceModule(B, links=switch_to_rules)

        # Start monitoring
        T = 1000  # The number of time steps
        self.trace = np.zeros((T, L))  # Time Step x Link ID
        self.trace2 = np.zeros((T, self.R))  # Time Step x Link ID
        for self.t in range(T):
            self.logger.info('Time Step: '+str(self.t))
            for dp in self.datapaths.values():  # Complete measurement
                self._request_stats(dp)  # See SimpleMonitor13._request_stats
            hub.sleep(1)

        # Save
        with open('trace_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.pkl', 'w') as trace_file:
            pickle.dump({
                'im'                : im,
                'im2'               : im2,
                'A'                 : A,
                'B'                 : B,
                'trace'             : self.trace,
                'trace2'            : self.trace2,
                'port_dst_to_rule'  : self.port_dst_to_rule,
                'switch_links'      : [(link.src.dpid,link.dst.dpid) for link in link_list],
                'host_links'        : [(host.mac,host.port.dpid) for host in host_list],
                'switch_to_links'   : switch_to_links,
                'switch_to_rules'   : switch_to_rules,
                'H'                 : H,
                'S'                 : S,
                'N'                 : N,
                'H2'                : H2,
                'L'                 : L,
                'R'                 : self.R,
                'T'                 : T
            }, trace_file)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        if self.t == None:
            return
        body = ev.msg.body
        switch_id = ev.msg.datapath.id

        for stat in sorted(body, key=attrgetter('port_no')):

            # Check if it is a switch link
            if stat.port_no in self.port_to_link[switch_id]:
                # Outgoing link
                link_id, link = self.port_to_link[switch_id][stat.port_no]
                self.trace[self.t, link_id] = stat.tx_packets

                # Incoming link
                link_id, link = self.port_to_link[
                    link.dst.dpid][link.dst.port_no]
                self.trace[self.t, link_id] = stat.rx_packets

            # Check if it is a host link
            elif switch_id in self.port_to_host and stat.port_no in self.port_to_host[switch_id]:
                rlink_id, tlink_id, host = self.port_to_host[
                    switch_id][stat.port_no]
                self.trace[self.t, rlink_id] = stat.rx_packets
                self.trace[self.t, tlink_id] = stat.tx_packets

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        switch_id = ev.msg.datapath.id
        for stat in sorted([flow for flow in body if flow.priority == 1],
                    key=lambda flow: (flow.match['in_port'],flow.match['eth_dst'])):
            if self.t == None:
                self.port_dst_to_rule.setdefault(switch_id,{})
                port_no = stat.instructions[0].actions[0].port
                self.port_dst_to_rule[switch_id][stat.match['in_port'],stat.match['eth_dst']] = self.R, port_no
                self.R += 1
            else:
                rule_id, port_no = self.port_dst_to_rule[switch_id][stat.match['in_port'],stat.match['eth_dst']]
                self.trace2[self.t, rule_id] = stat.packet_count

    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # Ignore lldp packets
        eth = packet.Packet(ev.msg.data).get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        super(Monitor, self)._packet_in_handler(ev)

