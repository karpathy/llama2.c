#ifndef LLAMA2CPP_TRANSFORMER_HPP
#define LLAMA2CPP_TRANSFORMER_HPP
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include "memory.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace llama2cpp {

/**
 * @brief Transformer configuration
 *
 */
struct TransformerConfig {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because
                     // of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
};

struct TransformerWeights {
    // token embedding table
    Tensor<CPU, float32_t> token_embedding_table;  // (vocab_size, dim)
    // weights for rmsnorms
    Tensor<CPU, float32_t> rms_att_weight;  // (layer, dim) rmsnorm weights
    Tensor<CPU, float32_t> rms_ffn_weight;   // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    Tensor<CPU, float32_t> wq;  // (layer, dim, n_heads * head_size)
    Tensor<CPU, float32_t> wk;  // (layer, dim, n_kv_heads * head_size)
    Tensor<CPU, float32_t> wv;  // (layer, dim, n_kv_heads * head_size)
    Tensor<CPU, float32_t> wo;  // (layer, n_heads * head_size, dim)
    // weights for ffn
    Tensor<CPU, float32_t> w1;  // (layer, hidden_dim, dim)
    Tensor<CPU, float32_t> w2;  // (layer, dim, hidden_dim)
    Tensor<CPU, float32_t> w3;  // (layer, hidden_dim, dim)
    // final rmsnorm
    Tensor<CPU, float32_t> rms_final_weight;  // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    Tensor<CPU, float32_t> wcls;
};

void memory_map_weights(TransformerWeights *weights, TransformerConfig &config, float *ptr, int shared_weights) {
    int head_size = config.dim / config.n_heads;
    // make sure the multiplications below are done in 64bit to fit the
    // parameter counts of 13B+ models
    unsigned long long n_layers = config.n_layers;

    // weights->token_embedding_table = ptr;
    weights->token_embedding_table.reShape(Shape(config.vocab_size, config.dim));
    weights->token_embedding_table.copyFrom(ptr, weights->token_embedding_table.numElements());
    ptr += config.vocab_size * config.dim;

    // weights->rms_att_weight = ptr;
    weights->rms_att_weight.reShape(Shape(config.n_layers, config.dim));
    weights->rms_att_weight.copyFrom(ptr, weights->rms_att_weight.numElements());
    ptr += n_layers * config.dim;

    // weights->wq = ptr;
    weights->wq.reShape(Shape(config.n_layers, config.dim, config.n_heads * head_size));
    weights->wq.copyFrom(ptr, weights->wq.numElements());
    ptr += n_layers * config.dim * (config.n_heads * head_size);

    // weights->wk = ptr;
    weights->wk.reShape(Shape(config.n_layers, config.dim, config.n_kv_heads * head_size));
    weights->wk.copyFrom(ptr, weights->wk.numElements());
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);

    // weights->wv = ptr;
    weights->wv.reShape(Shape(config.n_layers, config.dim, config.n_kv_heads * head_size));
    weights->wv.copyFrom(ptr, weights->wv.numElements());
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);

    // weights->wo = ptr;
    weights->wo.reShape(Shape(config.n_layers, config.dim, config.n_heads * head_size));
    weights->wo.copyFrom(ptr, weights->wo.numElements());
    ptr += n_layers * config.dim * (config.n_heads * head_size);

    // weights->rms_ffn_weight = ptr;
    weights->rms_ffn_weight.reShape(Shape(config.n_layers, config.dim));
    weights->rms_ffn_weight.copyFrom(ptr, weights->rms_ffn_weight.numElements());
    ptr += n_layers * config.dim;

    // weights->w1 = ptr;
    weights->w1.reShape(Shape(config.n_layers, config.hidden_dim, config.dim));
    weights->w1.copyFrom(ptr, weights->w1.numElements());
    ptr += n_layers * config.dim * config.hidden_dim;

    // weights->w2 = ptr;
    weights->w2.reShape(Shape(config.n_layers, config.dim, config.hidden_dim));
    weights->w2.copyFrom(ptr, weights->w2.numElements());
    ptr += n_layers * config.hidden_dim * config.dim;

    // weights->w3 = ptr;
    weights->w3.reShape(Shape(config.n_layers, config.hidden_dim, config.dim));
    weights->w3.copyFrom(ptr, weights->w3.numElements());
    ptr += n_layers * config.dim * config.hidden_dim;

    // weights->rms_final_weight = ptr;
    weights->rms_final_weight.reShape(Shape(config.dim));
    weights->rms_final_weight.copyFrom(ptr, weights->rms_final_weight.numElements());

    ptr += config.dim;
    ptr += config.seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    ptr += config.seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)

    // weights->wcls = shared_weights ? weights->token_embedding_table : ptr;
    weights->wcls.reShape(Shape(config.vocab_size, config.dim));
    auto data = shared_weights ? weights->token_embedding_table.data() : ptr;
    weights->wcls.copyFrom(data, weights->wcls.numElements());
}

void read_checkpoint(const std::string &checkpoint_path, TransformerConfig &config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
    FILE *file = fopen(checkpoint_path.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint_path.c_str());
        exit(EXIT_FAILURE);
    }
    // read in the config header
    if (fread(&config, sizeof(TransformerConfig), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    // negative vocab size is hacky way of signaling unshared weights. bit
    // yikes.
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END);  // move file pointer to end of file
    *file_size = ftell(file);  // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint_path.c_str(), O_RDONLY);  // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float *weights_ptr = *data + sizeof(TransformerConfig) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);

    // // C++
    // std::ifstream file(checkpoint_path, std::ios::binary);
    // if (!file)
    // {
    //     std::cerr << "Couldn't open file " << checkpoint_path << '\n';
    //     std::exit(EXIT_FAILURE);
    // }
    // file.read(reinterpret_cast<char *>(&config), sizeof(TransformerConfig));
    // // negative vocab size is hacky way of signaling unshared weights. bit
    // yikes. int shared_weights = config->vocab_size > 0 ? 1 : 0;
    // config->vocab_size = abs(config->vocab_size);
    // // figure out the file size
}

template <template <class> class COMPUTE, class T>
class Attention {
   public:
    using ptr = std::unique_ptr<Attention<COMPUTE, T>>;
    using value_type = T;
    using compute = COMPUTE<T>;

    explicit Attention(TensorView<float32_t> &wq, TensorView<float32_t> &wk, TensorView<float32_t> &wv, int kv_dim, int dim, int n_heads, int kv_heads,
                       int seq_len)
        : m_wq(wq),
          m_wk(wk),
          m_wv(wv),
          m_key_cache(Shape(seq_len * kv_dim)),
          m_value_cache(Shape(seq_len * kv_dim)),
          m_kv_dim(kv_dim),
          m_dim(dim),
          m_n_heads(n_heads),
          m_kv_heads(kv_heads),
          m_head_size(dim / n_heads),
          m_seq_len(seq_len),
          m_q(Shape(dim)),
          m_att(Shape(n_heads * seq_len)) {}

    void forward(Tensor<COMPUTE, value_type> &in, Tensor<COMPUTE, value_type> &xb, int pos_) {
        // in shape (dim), xb shape (dim)

        // key and value point to the kv cache
        TensorView<float32_t> k(m_key_cache.data() + pos_ * m_kv_dim, Shape(m_dim));
        TensorView<float32_t> v(m_value_cache.data() + pos_ * m_kv_dim, Shape(m_dim));

        // qkv matmuls for this position
        matmul(m_q, in, m_wq);
        matmul(k, in, m_wk);
        matmul(v, in, m_wv);

        // RoPE relative positional encoding: complex-valued rotate q and k in
        // each head
        for (int i = 0; i < m_dim; i += 2) {
            int head_dim = i % m_head_size;
            float32_t freq = 1.0f / powf(10000.0f, head_dim / static_cast<float32_t>(m_head_size));
            float32_t val = pos_ * freq;
            float32_t fcr = cosf(val);
            float32_t fci = sinf(val);
            int rotn = i < m_kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float32_t *vec = v == 0 ? m_q.data() : k.data();  // the vector to rotate (query or key)
                float32_t v0 = vec[i];
                float32_t v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        // @TODO write modular head implementation
        int kv_mul = m_n_heads / m_kv_heads;
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < m_n_heads; h++) {
            // get the query vector for this head
            TensorView<float32_t> q_(m_q.data() + h * m_head_size, Shape(m_head_size));
            // attention scores for this head
            TensorView<float32_t> att_(m_att.data() + h * m_seq_len, Shape(m_seq_len));
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos_; t++) {
                // get the key vector for this head and at this timestep
                // TODO use slice API instead
                TensorView<float32_t> k_(m_key_cache.data() + t * m_kv_dim + (h / kv_mul) * m_head_size, Shape(m_head_size));
                // calculate the attention score as the dot product of q and k
                float32_t score = dot_prod(q_, k_);
                score /= sqrtf(m_head_size);
                // save the score to the attention buffer
                att_(t) = score;
            }

            // softmax the scores to get attention weights, from 0..pos
            // inclusively
            softmax(att_.data(), pos_ + 1);

            // weighted sum of the values, store back into xb
            TensorView<float32_t> xb_(xb.data() + h * m_head_size, Shape(m_head_size));
            memset(xb_.data(), 0,
                   m_head_size * sizeof(float32_t));  //@TODO implement API to set value
            for (int t = 0; t <= pos_; t++) {
                // get the value vector for this head and at this timestep
                TensorView<float32_t> v_(m_value_cache.data() + t * m_kv_dim + (h / kv_mul) * m_head_size, Shape(m_head_size));
                // get the attention weight for this timestep
                float32_t a = att_[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < m_head_size; i++) {
                    xb_(i) += a * v_(i);
                }
            }
        }
    }

   private:
    TensorView<float32_t> m_wq;            // query (dim, n_heads * head_size)
    TensorView<float32_t> m_wk;            // key (dim, kv_dim * head_size)
    TensorView<float32_t> m_wv;            // value (dim, kv_dim * head_size)
    Tensor<CPU, float32_t> m_key_cache;    // key cache (seq_len * kv_dim)
    Tensor<CPU, float32_t> m_value_cache;  // value cache (seq_len * kv_dim)
    int m_kv_dim;                          // key value cache dimension ((dim * n_kv_heads) / n_heads)
    int m_dim;                             // transformer dimension
    int m_n_heads;                         // number of heads.
    int m_kv_heads;                        // number of key/value heads (can be < query heads because of multiquery)
    int m_head_size;                       // (dim / n_heads)
    int m_seq_len;                         // max sequence length
    Tensor<CPU, float32_t> m_q;            // query tensor
    Tensor<CPU, float32_t> m_att;          // attention tensor
};

template <template <class> class COMPUTE, class T>
class FeedForward {
   public:
    using ptr = std::unique_ptr<FeedForward<COMPUTE, T>>;
    using value_type = T;
    using compute = COMPUTE<T>;

    FeedForward(TensorView<float> &w1_, TensorView<float> &w2_, TensorView<float> &w3_, int dim, int hidden_dim)
        : m_dim(dim), m_hidden_dim(hidden_dim), m_w1(w1_), m_w2(w2_), m_w3(w3_), m_hb(Shape(hidden_dim)), m_hb2(Shape(hidden_dim)) {}

    /**
     * @brief forward pass for feedforward layer.
     *
     * self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
     */
    void forward(Tensor<COMPUTE, value_type> &in, Tensor<COMPUTE, value_type> &out) {
        matmul(m_hb, in, m_w1);
        matmul(m_hb2, in, m_w3);

        // SwiGLU non-linearity
        silu_inpl(m_hb);

        hadamard_prod(m_hb, m_hb, m_hb2);

        // final matmul to get the output of the ffn
        matmul(out, m_hb, m_w2);
    }

   private:
    int m_dim;  // transformer dimension.
    int m_hidden_dim;
    TensorView<float32_t> m_w1;    // (hidden_dim, dim)
    TensorView<float32_t> m_w2;    // (hidden_dim, dim)
    TensorView<float32_t> m_w3;    // (hidden_dim, dim)
    Tensor<CPU, float32_t> m_hb;   // buffer for hidden dimension (hidden_dim,)
    Tensor<CPU, float32_t> m_hb2;  // buffer for hidden dimension (hidden_dim,)
};

template <template <class> class COMPUTE, class T>
class TransformerBlock {
   public:
    using ptr = std::unique_ptr<TransformerBlock<COMPUTE, T>>;
    using value_type = T;
    using compute = COMPUTE<T>;

    explicit TransformerBlock(Attention<COMPUTE, value_type>::ptr attention, FeedForward<COMPUTE, value_type>::ptr feed_forward,
                              TensorView<float> &rms_ffn_weight, TensorView<float32_t> &wo, TensorView<float32_t> &w_rms_att, int dim)
        : m_attention(std::move(attention)),
          m_feedforward(std::move(feed_forward)),
          m_w_rms_ffn(rms_ffn_weight),
          m_xh(Shape(dim)),
          m_xh2(Shape(dim)),
          m_wo(wo),
          m_w_rms_att(w_rms_att) {}

    void forward(Tensor<COMPUTE, value_type> &x, int pos_) {
        // attention rmsnorm
        rmsnorm(m_xh, x, m_w_rms_att);

        // forward attention.
        m_attention->forward(m_xh, m_xh, pos_);

        // final matmul to get the output of the attention
        matmul(m_xh2, m_xh, m_wo);

        // residual connection back into x
        add(x, x, m_xh2);

        // ffn rmsnorm
        rmsnorm(m_xh, x, m_w_rms_ffn);

        // forward FFN.
        m_feedforward->forward(m_xh, m_xh);

        // residual connection
        add(x, x, m_xh);
    }

   private:
    Attention<COMPUTE, value_type>::ptr m_attention;
    FeedForward<COMPUTE, value_type>::ptr m_feedforward;
    // unsigned long long m_l;
    Tensor<COMPUTE, value_type> m_xh;   // hidden state same a x but used inside branch
    Tensor<COMPUTE, value_type> m_xh2;  // hidden state2 same a x but used inside branch
    // weights for matmuls. note dim == n_heads * head_size
    TensorView<value_type> m_wo;         // (n_heads * head_size, dim)
    TensorView<value_type> m_w_rms_att;  // (dim)
    TensorView<value_type> m_w_rms_ffn;  // (dim)
};

template <template <class> class COMPUTE, class T>
class Linear {
   public:
    using ptr = typename std::unique_ptr<Linear<COMPUTE, T>>;
    using compute = COMPUTE<T>;

    Linear(TensorView<T> &wcls) : m_wcls(wcls) {}

    void forward(const Tensor<CPU, T> &x, Tensor<CPU, T> &out) { matmul(out, x, m_wcls); }

   private:
    TensorView<T> m_wcls;  // classification weights (out_dim, in_dim)
};

class Transformer {
   public:
    using ptr = std::unique_ptr<Transformer>;
    explicit Transformer(const std::string &checkpoint_path) {
        // read in the Config and the Weights from the checkpoint
        read_checkpoint(checkpoint_path, m_config, &m_weights, &m_fd, &m_data, &m_file_size);

        initializeLayers();
    }

    void initializeLayers() {
        int kv_dim = (m_config.dim * m_config.n_kv_heads) / m_config.n_heads;
        int dim = m_config.dim;
        auto n_heads = m_config.n_heads;
        int head_size = m_config.dim / m_config.n_heads;
        int hidden_dim = m_config.hidden_dim;

        for (unsigned long long l = 0; l < m_config.n_layers; l++) {
            // Attention layer
            int loff_ = l * m_config.seq_len * kv_dim;  // kv cache layer offset for convenience
            TensorView<float32_t> wq(m_weights.wq.data() + l * m_config.dim * m_config.dim, Shape(dim, n_heads * head_size));
            TensorView<float32_t> wk(m_weights.wk.data() + l * m_config.dim * kv_dim, Shape(dim, kv_dim * head_size));
            TensorView<float32_t> wv(m_weights.wv.data() + l * m_config.dim * kv_dim, Shape(dim, kv_dim * head_size));

            auto attention = std::make_unique<Attention<CPU, float32_t>>(wq, wk, wv, kv_dim, dim, m_config.n_heads, m_config.n_kv_heads, m_config.seq_len);

            // FF layer
            TensorView<float32_t> w1(m_weights.w1.data() + l * dim * hidden_dim, Shape(hidden_dim, dim));
            TensorView<float32_t> w2(m_weights.w2.data() + l * dim * hidden_dim, Shape(dim, hidden_dim));
            TensorView<float32_t> w3(m_weights.w3.data() + l * dim * hidden_dim, Shape(hidden_dim, dim));
            auto feedforward = std::make_unique<FeedForward<CPU, float32_t>>(w1, w2, w3, dim, m_config.hidden_dim);

            TensorView<float32_t> wo(m_weights.wo.data() + l * dim * dim, Shape(dim, dim));
            TensorView<float32_t> w_rms_att(m_weights.rms_att_weight.data() + l * dim, Shape(dim));
            TensorView<float32_t> w_rms_ffn(m_weights.rms_ffn_weight.data() + l * dim, Shape(dim));
            m_layers.push_back(std::make_unique<TransformerBlock<CPU, float32_t>>(std::move(attention), std::move(feedforward), w_rms_ffn, wo, w_rms_att, dim));
        }

        m_linear = std::make_unique<Linear<CPU, float32_t>>(m_weights.wcls);
    }

    ~Transformer() {
        // close the memory mapping
        if (m_data != MAP_FAILED) {
            munmap(m_data, m_file_size);
        }
        if (m_fd != -1) {
            close(m_fd);
        }
    }

    void forward(int token, int pos, Tensor<CPU, float32_t> &logits) {
        // a few convenience variables
        TransformerWeights *w = &m_weights;
        size_t dim = static_cast<size_t>(m_config.dim);

        // copy the token embedding into x
        TensorView<float32_t> content_row(w->token_embedding_table.data() + token * dim, Shape(dim));

        Tensor<CPU, float32_t> x_in(content_row.data(), Shape(dim));

        // forward all the layers
        for (auto &layer : m_layers) {
            layer->forward(x_in, pos);
        }

        // final rmsnorm
        rmsnorm(x_in, x_in, w->rms_final_weight);

        // classifier into logits
        m_linear->forward(x_in, logits);
    }

    auto getConfig() const -> const TransformerConfig & { return m_config; }

   private:
    TransformerConfig m_config;    // the hyperparameters of the architecture (the blueprint)
    TransformerWeights m_weights;  // the weights of the model
    // some more state needed to properly clean up the memory mapping (sigh)
    int m_fd;             // file descriptor for memory mapping
    float *m_data;        // memory mapped data pointer
    ssize_t m_file_size;  // size of the checkpoint file in bytes
    Linear<CPU, float32_t>::ptr m_linear;
    std::vector<TransformerBlock<CPU, float32_t>::ptr> m_layers;
};

}  // namespace llama2cpp
#endif