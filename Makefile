# Contains helpful targets for building and running the project.

all: build

.PHONY: build
build: run

run: run.c
	gcc -O3 -o run run.c -lm

.PHONY: run-inference
run-inference: .venv run out/model.bin
	.venv/bin/python run_wrap.py

out/model.bin:
	wget https://karpathy.ai/llama2c/model.bin -P out

.venv:
	python3 -m venv .venv
	.venv/bin/pip install wheel
	.venv/bin/pip install sentencepiece

.PHONY: clean
clean:
	rm -f run
	rm -rf .venv

.PHONY: realclean
realclean:
	rm -rf out
