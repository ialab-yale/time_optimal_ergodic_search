#!/bin/zsh

for i in {30..39}; do
    python receding_horizonDubinsCar.py -s $i -th 5 -trial $i
done