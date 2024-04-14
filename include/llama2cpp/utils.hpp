#ifndef LLAMA2CPP_UTILS_HPP
#define LLAMA2CPP_UTILS_HPP
#include <string>
#include <fstream>
#include "transformer/transformer.hpp"

namespace llama2cpp {

void loadModel(const std::string &checkpoint_path, TransformerConfig &config, TransformerWeights<CPU, float32_t> &weights) {
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file) {
        std::cerr << "Couldn't open file " << checkpoint_path << '\n';
        std::exit(EXIT_FAILURE);
    }
    file.read(reinterpret_cast<char *>(&config), sizeof(TransformerConfig));
    auto shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = std::abs(config.vocab_size);

    size_t dim = static_cast<size_t>(config.dim);
    size_t head_size = static_cast<size_t>(config.dim / config.n_heads);
    size_t vocab_size = static_cast<size_t>(config.vocab_size);
    size_t n_heads = static_cast<size_t>(config.n_heads);
    size_t n_kv_heads = static_cast<size_t>(config.n_kv_heads);
    size_t hidden_dim = static_cast<size_t>(config.hidden_dim);

    // make sure the multiplications below are done in 64bit to fit the
    // parameter counts of 13B+ models
    unsigned long long n_layers = config.n_layers;

    weights.token_embedding_table.reShape(Shape(vocab_size, dim));
    file.read(reinterpret_cast<char *>(weights.token_embedding_table.data()), weights.token_embedding_table.numBytes());

    weights.rms_att_weight.reShape(Shape(n_layers, dim));
    file.read(reinterpret_cast<char *>(weights.rms_att_weight.data()), weights.rms_att_weight.numBytes());

    weights.wq.reShape(Shape(n_layers, dim, n_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wq.data()), weights.wq.numBytes());

    weights.wk.reShape(Shape(n_layers, dim, n_kv_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wk.data()), weights.wk.numBytes());

    weights.wv.reShape(Shape(n_layers, dim, n_kv_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wv.data()), weights.wv.numBytes());

    weights.wo.reShape(Shape(n_layers, dim, n_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wo.data()), weights.wo.numBytes());

    weights.rms_ffn_weight.reShape(Shape(n_layers, dim));
    file.read(reinterpret_cast<char *>(weights.rms_ffn_weight.data()), weights.rms_ffn_weight.numBytes());

    weights.w1.reShape(Shape(n_layers, hidden_dim, dim));
    file.read(reinterpret_cast<char *>(weights.w1.data()), weights.w1.numBytes());

    weights.w2.reShape(Shape(n_layers, dim, hidden_dim));
    file.read(reinterpret_cast<char *>(weights.w2.data()), weights.w2.numBytes());

    weights.w3.reShape(Shape(n_layers, hidden_dim, dim));
    file.read(reinterpret_cast<char *>(weights.w3.data()), weights.w3.numBytes());

    weights.rms_final_weight.reShape(Shape(dim));
    file.read(reinterpret_cast<char *>(weights.rms_final_weight.data()), weights.rms_final_weight.numBytes());

    // ptr += config.dim;
    // ptr += config.seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    // ptr += config.seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)
    weights.wcls.reShape(Shape(vocab_size, dim));
    if (!shared_weights) {
        file.seekg((config.dim + config.seq_len * head_size) * sizeof(float32_t), std::ios::cur);
        file.read(reinterpret_cast<char *>(weights.wcls.data()), weights.wcls.numBytes());
    } else {
        weights.wcls.copyFrom(weights.token_embedding_table.data(), weights.wcls.numElements());
    }

    file.close();
}

void loadModel(const std::string &checkpoint_path, TransformerConfig &config, TransformerWeights<CUDA, float32_t> &weights){
    //@TODO implement this
}


}  // namespace llama2cpp
#endif