for i in `seq 1 $1`
do
    ovs-vsctl set bridge s$i protocol=OpenFlow13
done
