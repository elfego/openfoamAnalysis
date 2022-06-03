#!/bin/bash

for i in $(seq 0 3); do
    ./main.py $1 --range `echo "$(($i * 100)):$(($i * 100 + 99))"` &
done

wait

echo "Done post-processing the simulation $1"
