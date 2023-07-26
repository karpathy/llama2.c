CC=gcc
OPT=-O3

FAST_OPT=-Ofast

FAST_OMP=$(FAST_OPT) -fopen-mp -march=native


# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) $(OPT) -o run run.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin 1.0 3
rundebug: run.c
	$(CC) -g -o run run.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations. 
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs. 
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific 
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens. 
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) $(FAST_OPT) -o run run.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) $(FAST_OMP) run.c  -lm  -o run

.PHONY: clean
clean:
	rm -f run
