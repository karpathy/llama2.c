#!/usr/bin/env bash

if [ -d "/usr/include/openblas" ]; then
    gcc -DUSE_OPENBLAS -I/usr/include/openblas -Ofast -o run run.c -lm -lopenblas
else
    gcc -Ofast -o run run.c -lm
fi
