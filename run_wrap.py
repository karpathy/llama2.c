"""
wrapper around run.c
mostly deals with the sentencepiece encoding/decoding
C code does all the transformer inference of the individual tokens
"""

from tokenizer import Tokenizer
import subprocess
import time

# specify your command
command = ["./run", "model.bin", "0.0"]

# Start the process
proc = subprocess.Popen(command, stdout=subprocess.PIPE)
enc = Tokenizer()

t0 = time.time()
tokens = []
for line in proc.stdout:
    token = int(line.decode('utf-8').strip())
    dec = enc.decode([token])
    print(dec, end=" ", flush=True)
    tokens.append(token)
t1 = time.time()

print('\n---\n')
print("Sorry I'm not sure why sentencepiece can't stream tokens properly, I'll solve it later. Here is the whole thing:")
print('\n---\n')
print(enc.decode(tokens))

print(f"achieved tok/s: {len(tokens) / (t1 - t0)}")
# Wait for the process to finish
proc.wait()
