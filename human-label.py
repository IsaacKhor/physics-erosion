#!/usr/bin/env python3

import sys

lbf = open('labels.txt', 'w')
pyf = open('pylabels.txt', 'w')
uif = open('userin.txt', 'w')

labels = []
sim_no = 1

while True:
    inp = sys.stdin.readline().strip()

    if inp == 'exit':
        break

    uif.write(inp)

    #if inp[0] == 'i':
    #    new_sim_no = int(inp[1:], 10)
    #    print('Setting sim_no to', new_sim_no)
    #    sim_no = new_sim_no
    #    continue

    for c in inp:
        if c == 'h':
            labels.append([1,0])
            lbf.write('[1,0],\n')
            print('Sim', sim_no, ': left')
        elif c == 's':
            labels.append([0,1])
            lbf.write('[0,1],\n')
            print('Sim', sim_no, ': right')
        elif c == 'c':
            labels.append([0,0])
            lbf.write('[0,0],\n')
            print('Sim', sim_no, ': centre/neither')
        elif c == 'b':
            labels.append([1,1])
            lbf.write('[1,1],\n')
            print('Sim', sim_no, ': both left/right')
        else:
            print('Unknown input')
            continue
        sim_no += 1

pyf.write(str(labels))

lbf.close()
pyf.close()
uif.close()