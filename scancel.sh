#!/bin/bash
for index in $(seq 517511 517551); do
    echo "Cancelling $index"
    scancel $index -u ddy19-exk01
done

squeue -u ddy19-exk01
