#!/usr/bin/env bash

#gcc -DUSE_OPENBLAS -I/usr/include/openblas -Ofast -o run run.c -lm -lopenblas
gcc -DUSE_OPENMP -Ofast -o run run.c -lm -fopenmp
