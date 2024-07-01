# llama2.sycl.cpp

llama2.sycl.cpp is the SCYL version of [llama2.c](https://github.com/karpathy/llama2.c).

# Requirements
- icpx compiler which is included in oneAPI Base Toolkit available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
- model checkpoint needs to be downloaded from [huggingface](https://huggingface.co/meta-llama/Llama-2-7b/tree/main) and converted using a Python script available [here](https://github.com/karpathy/llama2.c). That same page also has the tokenizer.bin file.

# Build Instructions
First, source icpx compiler. Then,

```
make
```
This creates the executable `runsycl`

# Run Instructions

```
./runsycl <converted check_point> -z <tokenizer_path> -m generate -t 0.0 -n <num_tot_tokens> -i <prompt>
```
