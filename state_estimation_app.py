from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

class StateEstimationApp(app_manager.RyuApp):
    """Currently, just a sample code taken from Internet and modified a bit. There is nothing about state estimation at this point"""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(StateEstimationApp, self).__init__(*args, **kwargs)

    def send_desc_stats_requests(self, datapath):
        ofp_parser = datapath.ofproto_parser

        req = ofp_parser.OFPDescStatsRequest(datapath,0)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPDescStatsReply, MAIN_DISPATCHER)
    def desc_stats_reply_handler(self,ev):    
        body = ev.msg.body

        serial_num = body.serial_num if body.serial_num else 'None'
        dp_desc = body.dp_desc if body.dp_desc else 'None'
        print'OFPDescStatsReply received: mfr_desc=%s hw_desc=%s sw_desc=%s ' % ( body.mfr_desc, body.hw_desc, body.sw_desc)

    def add_flow(self, datapath, in_port, dst, actions):
        ofproto = datapath.ofproto

        match = datapath.ofproto_parser.OFPMatch(
            in_port=in_port, dl_dst=haddr_to_bin(dst))

        mod = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=match, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
            priority=ofproto.OFP_DEFAULT_PRIORITY,
            flags=ofproto.OFPFF_SEND_FLOW_REM, actions=actions)

        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port = {}
        self.mac_to_port.setdefault(dpid, {})

        #self.logger.info("packet in %s %s %s %s", dpid, src, dst, msg.match['in_port'])
        print "packet in %s %s %s %s" %(dpid, src, dst, msg.match['in_port'])

        self.send_desc_stats_requests(datapath)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = msg.match['in_port']

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            self.add_flow(datapath, msg.match['in_port'], dst, actions)

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id, in_port=msg.match['in_port'],
            actions=actions)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def multipart_reply_handler(self, ev):
        msg = ev.msg
        reason = msg.reason
        port_no = msg.desc.port_no

        ofproto = msg.datapath.ofproto
        if reason == ofproto.OFPPR_ADD:
            print("port added %s" %port_no)
        elif reason == ofproto.OFPPR_DELETE:
            print("port deleted %s" % port_no)
        elif reason == ofproto.OFPPR_MODIFY:
            print("port modified %s" % port_no)
        else:
            print("Illegal port state %s %s" % (port_no, reason))
