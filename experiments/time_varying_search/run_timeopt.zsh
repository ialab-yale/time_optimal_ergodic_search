#!/bin/zsh

for i in {11..20}; do
    python receding_horizonDubinsCarTimeOpt.py -s $i -trial $i
done