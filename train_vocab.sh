#!/bin/bash

# Trains a sentencepiece tokenizer model on a bunch of given data, my best
# effort attempt to replicate how Meta trained their Llama 2 tokenizer.

# usage: $ train_vocab.sh <input> <model_prefix> <vocab_size>
# example:
# ./train_vocab.sh tiny.txt tokenizer_tiny 1024
# requirements:
# install https://github.com/google/sentencepiece

# check if the correct number of arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <input> <model_prefix> <vocab_size>"
    exit 1
fi

# assign command-line arguments to variables
input=$1
model_prefix=$2
vocab_size=$3

# check if input file exists
if [ ! -f "$input" ]; then
    echo "Usage: $0 <input> <model_prefix> <vocab_size>"
    echo "input '$input' not found."
    exit 1
fi

# check if vocab_size is a positive integer
if ! [[ "$vocab_size" =~ ^[0-9]+$ ]] || [ "$vocab_size" -lt 1 ]; then
    echo "Usage: $0 <input> <model_prefix> <vocab_size>"
    echo "vocab_size size must be a positive integer."
    exit 1
fi

# Print the processed inputs
echo "Input: $input"
echo "Model Prefix: $model_prefix"
echo "Vocabulary Size: $vocab_size"

# train a sentencepiece tokenizer model
# Llama 2 config can be printed as follows:

# import sentencepiece.sentencepiece_model_pb2
# mp = sentencepiece.sentencepiece_model_pb2.ModelProto()
# mp.ParseFromString(open("tokenizer.model", "rb").read())
# print(mp.trainer_spec)
# print(mp.normalizer_spec)

# this gives:

# trainer_spec {
#   input: "/large_experiments/theorem/datasets/MERGED/all.test1.merged"
#   model_prefix: "spm_model_32k_200M_charcov099995_allowWSO__v2"
#   model_type: BPE
#   vocab_size: 32000
#   self_test_sample_size: 0
#   input_format: "text"
#   character_coverage: 0.9999499917030334
#   input_sentence_size: 200000000
#   seed_sentencepiece_size: 1000000
#   shrinking_factor: 0.75
#   num_threads: 80
#   num_sub_iterations: 2
#   max_sentence_length: 4192
#   shuffle_input_sentence: true
#   max_sentencepiece_length: 16
#   split_by_unicode_script: true
#   split_by_whitespace: true
#   split_by_number: true
#   treat_whitespace_as_suffix: false
#   split_digits: true
#   allow_whitespace_only_pieces: true
#   vocabulary_output_piece_score: true
#   hard_vocab_limit: true
#   use_all_vocab: false
#   byte_fallback: true
#   required_chars: ""
#   unk_id: 0
#   bos_id: 1
#   eos_id: 2
#   pad_id: -1
#   unk_surface: " \342\201\207 "
#   unk_piece: "<unk>"
#   bos_piece: "<s>"
#   eos_piece: "</s>"
#   pad_piece: "<pad>"
#   train_extremely_large_corpus: false
#   enable_differential_privacy: false
#   differential_privacy_noise_level: 0.0
#   differential_privacy_clipping_threshold: 0
# }
# normalizer_spec {
#   name: "identity"
#   precompiled_charsmap: ""
#   add_dummy_prefix: true
#   remove_extra_whitespaces: false
#   normalization_rule_tsv: ""
# }

# let's now use spm_train to train this exact model
# options docs: https://github.com/google/sentencepiece/blob/master/doc/options.md

# we'll depart on a few settings:
# character_coverage -> 1.0

# other important notes:
# --split-digits = true, per the paper
# --allow_whitespace_only_pieces is true, default in spm is false
# --byte_fallback is true, default in spm is false
# --normalization_rule_name is identity, default in spm is nmt_nfkc

spm_train --input="$input" \
          --model_prefix="$model_prefix" \
          --model_type=bpe \
          --vocab_size="$vocab_size" \
          --self_test_sample_size=0 \
          --input_format="text" \
          --character_coverage=1.0 \
          --num_threads="$(nproc)" \
          --split_digits=true \
          --allow_whitespace_only_pieces=true \
          --byte_fallback=true \
          --unk_surface=" \342\201\207 " \
          --normalization_rule_name=identity \
