#!/bin/bash

for i in $(seq 0 7); do
    ./main.py $1 --range `echo "$(($i * 50)):$(($i * 50 + 50))"` &
done

wait

echo "Done post-processing the simulation $1"
