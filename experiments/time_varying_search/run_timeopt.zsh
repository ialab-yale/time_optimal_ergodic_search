#!/bin/zsh

for i in {0..29}; do
    python receding_horizonDubinsCarTimeOpt.py -s $i -trial $i
done