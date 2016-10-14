#!/usr/bin/env python

"""
quantize pcap file in time domain slots and write (n, 2) npy array
 [packets, bytes] for each row
"""

import sys
import pdb
from tqdm import tqdm
import numpy as np
import argparse
import dpkt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def xopen(fn, mode):
    if fn == '-':
        if mode.startswith('r'):
            return sys.stdin
        elif mode.startswith('w'):
            return sys.stdout
        else:
            raise ValueError("mode should be r or w")
    else:
        return open(fn, mode)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=float, default=1.0, help="granularity in seconds (millisecond precision)")
    parser.add_argument('-n', type=int, default=-1, help="number of packets to be parsed (-1 for inifinite)")
    parser.add_argument('--plot', type=str, default=None, help="plot output")
    parser.add_argument('pcapfn', type=str, help="input pcap filename (- for stdin)", default="-")
    parser.add_argument('outfn', type=str, help="output filename", default="f")

    args = parser.parse_args()

    outfn = args.outfn
    if not outfn.endswith(".npy"):
        outfn += ".npy"

    g_ms = int(args.g*1000)
    timeslots = {}
    sid_min = sys.maxint
    sid_max = 0
    invalid_cnt = 0
    n_p = 0
    with xopen(args.pcapfn, 'rb') as f_in:
        pcap_in = dpkt.pcap.Reader(f_in)
        t = args.n if args.n >= 0 else None
        for timestamp, buf in tqdm(pcap_in, total=t, ascii=True, unit_scale=True, unit='p'):
            eth = dpkt.ethernet.Ethernet(buf)
            if eth.type == 0x0800:
                plen = eth.data.len
                sid = int(timestamp*1000)/g_ms
                sid_min = min(sid_min, sid)
                sid_max = max(sid_max, sid)
                timeslots.setdefault(sid, np.array([0, 0]))
                timeslots[sid] += (1, plen)
            else:
                invalid_cnt += 1
            n_p += 1
            if args.n > 0 and n_p >= args.n:
                break
                
    #
    if invalid_cnt > 0:
        sys.stderr.write("there was {} packets ignored.\n".format(invalid_cnt))
    n = (sid_max - sid_min)+1
    x = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        sid = sid_min + i
        if sid in timeslots:
            x[i] = timeslots[sid]
    np.save(outfn, x)
    sys.stderr.write("{} slots written to {}\n".format(n, outfn))
    if args.plot != None:
        plt.plot(x[:,0], label="packets")
        plt.plot(x[:,1], label="bytes")
        plt.legend()
        plt.savefig(args.plot)
        sys.stderr.write("plotted in {}\n".format(args.plot))
            
            
