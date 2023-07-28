how to run PoC

```bash
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

adb push tokenizer.bin /data/local/tmp/tokenizer.bin
adb push stories15M.bin /data/local/tmp/stories15M.bin

adb shell chmod 777 /data/local/tmp/stories15M.bin
adb shell chmod 777 /data/local/tmp/tokenizer.bin
```