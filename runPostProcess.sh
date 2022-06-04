#!/bin/bash

for i in $(seq 0 9); do
    ./main.py $1 --range `echo "$(($i * 40)):$(($i * 40 + 39))"` &
done

wait

echo "Done post-processing the simulation $1"
