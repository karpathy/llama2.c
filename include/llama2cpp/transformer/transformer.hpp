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
#include "types.hpp"

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
template <template <class> class COMPUTE, class T>
struct TransformerWeights {
    // token embedding table
    Tensor<COMPUTE, T> token_embedding_table;  // (vocab_size, dim)
    // weights for rmsnorms
    Tensor<COMPUTE, T> rms_att_weight;  // (layer, dim) rmsnorm weights
    Tensor<COMPUTE, T> rms_ffn_weight;  // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    Tensor<COMPUTE, T> wq;  // (layer, dim, n_heads * head_size)
    Tensor<COMPUTE, T> wk;  // (layer, dim, n_kv_heads * head_size)
    Tensor<COMPUTE, T> wv;  // (layer, dim, n_kv_heads * head_size)
    Tensor<COMPUTE, T> wo;  // (layer, n_heads * head_size, dim)
    // weights for ffn
    Tensor<COMPUTE, T> w1;  // (layer, hidden_dim, dim)
    Tensor<COMPUTE, T> w2;  // (layer, dim, hidden_dim)
    Tensor<COMPUTE, T> w3;  // (layer, hidden_dim, dim)
    // final rmsnorm
    Tensor<COMPUTE, T> rms_final_weight;  // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    Tensor<COMPUTE, T> wcls;  // (vocab_size, dim)
};

template <template <class> class COMPUTE, class T>
class Attention {
   public:
    using ptr = typename std::unique_ptr<Attention<COMPUTE, T>>;
    using value_type = T;
    using compute = COMPUTE<T>;

    explicit Attention(TensorView<value_type> &wq, TensorView<value_type> &wk, TensorView<value_type> &wv, size_t kv_dim, size_t dim, size_t n_heads,
                       size_t kv_heads, size_t seq_len)
        : m_wq(wq),
          m_wk(wk),
          m_wv(wv),
          m_key_cache(Shape(seq_len * kv_dim)),
          m_value_cache(Shape(seq_len * kv_dim)),
          m_q(Shape(dim)),
          m_att(Shape(n_heads, seq_len)),
          m_kv_dim(kv_dim),
          m_dim(dim),
          m_n_heads(n_heads),
          m_kv_heads(kv_heads),
          m_head_size(dim / n_heads),
          m_seq_len(seq_len) {}

    void forward(Tensor<COMPUTE, value_type> &in, Tensor<COMPUTE, value_type> &xb, int pos_) {
        // in shape (dim), xb shape (dim)

        // key and value point to the kv cache
        // Note kv_cache is (seq_len*kv_dim) i.e (seq_len*(dim*n_kv_heads/n_heads))
        TensorView<value_type> k = m_key_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos_);
        TensorView<value_type> v = m_value_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos_);

        // qkv matmuls for this position
        matmul(m_q, in, m_wq);
        matmul(k, in, m_wk);
        matmul(v, in, m_wv);

        // RoPE relative positional encoding: complex-valued rotate q and k in
        // each head
        // Currently CPU only
        for (size_t i = 0; i < m_dim; i += 2) {
            size_t head_dim = i % m_head_size;
            float32_t freq = 1.0f / powf(10000.0f, head_dim / static_cast<float32_t>(m_head_size));
            float32_t val = pos_ * freq;
            float32_t fcr = cosf(val);
            float32_t fci = sinf(val);
            size_t rotn = i < m_kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
            for (size_t v = 0; v < rotn; v++) {
                TensorView<value_type> vec = v == 0 ? m_q : k;  // the vector to rotate (query or key)
                value_type v0 = vec[i];
                value_type v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        size_t kv_mul = m_n_heads / m_kv_heads;
        size_t h;
#pragma omp parallel for private(h)
        for (h = 0; h < m_n_heads; h++) {
            // get the query vector for this head
            TensorView<value_type> q_ = m_q.view(Shape(m_n_heads, m_head_size)).slice(h);

            // attention scores for this head
            TensorView<value_type> att_ = m_att.slice(h);

            // iterate over all timesteps, including the current one
            for (size_t t = 0; t <= pos_; t++) {
                // get the key vector for this head and at this timestep
                // head_size = (dim / n_heads)
                // kv_cache is (seq_len * kv_dim)
                //          -> (seq_len * (dim * n_kv_heads / n_heads)) 
                //          -> (seq_len*n_kv_heads, dim/n_heads) 
                //          -> (seq_len*n_kv_heads, head_size)
                // offset = t * m_kv_dim + (h / kv_mul) 
                //          -> t * (dim * n_kv_heads / n_heads) + h * n_kv_heads / n_heads 
                //          -> (t * dim + h)*(n_kv_heads / n_heads)
                TensorView<value_type> k_(m_key_cache.data() + t * m_kv_dim + (h / kv_mul) * m_head_size, Shape(m_head_size));
                // calculate the attention score as the dot product of q and k
                value_type score = dot_prod(q_, k_);
                score /= sqrtf(m_head_size);
                // save the score to the attention buffer
                att_(t) = score;
            }

            // softmax the scores to get attention weights, from 0..pos
            // inclusively
            softmax(att_, pos_ + 1);

            // weighted sum of the values, store back into xb
            // view xb of shape (dim) as (n_heads * head_size)
            TensorView<value_type> xb_ = xb.view(Shape(m_n_heads, m_head_size)).slice(h);

            setZero(xb_);
            for (size_t t = 0; t <= pos_; t++) {
                // get the value vector for this head and at this timestep
                TensorView<value_type> v_(m_value_cache.data() + t * m_kv_dim + (h / kv_mul) * m_head_size, Shape(m_head_size));
                // get the attention weight for this timestep
                value_type a = att_[t];
                // accumulate the weighted value into xb
                for (size_t i = 0; i < m_head_size; i++) {
                    xb_(i) += a * v_(i);
                }
            }
        }
    }

   private:
    TensorView<value_type> m_wq;                // query (dim, n_heads * head_size)
    TensorView<value_type> m_wk;                // key (dim, kv_dim * head_size)
    TensorView<value_type> m_wv;                // value (dim, kv_dim * head_size)
    Tensor<COMPUTE, value_type> m_key_cache;    // key cache (seq_len * kv_dim)
    Tensor<COMPUTE, value_type> m_value_cache;  // value cache (seq_len * kv_dim)
    Tensor<COMPUTE, value_type> m_q;            // query tensor (dim)
    Tensor<COMPUTE, value_type> m_att;          // attention tensor (n_heads * seq_len)
    size_t m_kv_dim;                            // key value cache dimension ((dim * n_kv_heads) / n_heads)
    size_t m_dim;                               // transformer dimension
    size_t m_n_heads;                           // number of heads.
    size_t m_kv_heads;                          // number of key/value heads (can be < query heads because of multiquery)
    size_t m_head_size;                         // (dim / n_heads)
    size_t m_seq_len;                           // max sequence length
};

template <template <class> class COMPUTE, class T>
class FeedForward {
   public:
    using ptr = std::unique_ptr<FeedForward<COMPUTE, T>>;
    using value_type = T;
    using compute = COMPUTE<T>;

    FeedForward(TensorView<value_type> &w1_, TensorView<value_type> &w2_, TensorView<value_type> &w3_, size_t dim, size_t hidden_dim)
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
    size_t m_dim;                       // transformer dimension.
    size_t m_hidden_dim;                // hidden layer dimension.
    TensorView<value_type> m_w1;        // (hidden_dim, dim)
    TensorView<value_type> m_w2;        // (hidden_dim, dim)
    TensorView<value_type> m_w3;        // (hidden_dim, dim)
    Tensor<COMPUTE, value_type> m_hb;   // buffer for hidden dimension (hidden_dim,)
    Tensor<COMPUTE, value_type> m_hb2;  // buffer for hidden dimension (hidden_dim,)
};

template <template <class> class COMPUTE, class T>
class TransformerBlock {
   public:
    using ptr = typename std::unique_ptr<TransformerBlock<COMPUTE, T>>;
    using value_type = T;
    using compute = COMPUTE<T>;

    explicit TransformerBlock(Attention<COMPUTE, value_type>::ptr attention, FeedForward<COMPUTE, value_type>::ptr feed_forward,
                              TensorView<value_type> &rms_ffn_weight, TensorView<value_type> &wo, TensorView<value_type> &w_rms_att, size_t dim)
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
    using value_type = T;
    using compute = COMPUTE<T>;

    Linear(TensorView<value_type> &wcls) : m_wcls(wcls) {}

    void forward(const Tensor<COMPUTE, value_type> &x, Tensor<COMPUTE, value_type> &out) { matmul(out, x, m_wcls); }

    auto outDim() const -> const size_t { return m_wcls.shape().shapeVec()[0]; }

   private:
    TensorView<value_type> m_wcls;  // classification weights (out_dim, in_dim)
};

template <template <class> class COMPUTE, class T>
class Transformer {
   public:
    using ptr = std::unique_ptr<Transformer>;
    using value_type = T;
    using compute = COMPUTE<T>;

    Transformer(TransformerConfig &config, TransformerWeights<COMPUTE, value_type> &weights) : m_config(config), m_weights(weights), m_linear(nullptr) {
        initializeLayers();
    }

    void initializeLayers() {
        size_t kv_dim = static_cast<size_t>((m_config.dim * m_config.n_kv_heads) / m_config.n_heads);
        size_t dim = static_cast<size_t>(m_config.dim);
        size_t n_heads = static_cast<size_t>(m_config.n_heads);
        size_t head_size = static_cast<size_t>(m_config.dim / m_config.n_heads);
        size_t hidden_dim = static_cast<size_t>(m_config.hidden_dim);
        size_t n_kv_heads = static_cast<size_t>(m_config.n_kv_heads);
        size_t seq_len = static_cast<size_t>(m_config.seq_len);

        // NOTE dim == n_heads * head_size

        for (size_t layer_idx = 0; layer_idx < m_config.n_layers; layer_idx++) {
            // Attention layer
            TensorView<value_type> wq = m_weights.wq.slice(layer_idx);  // (dim, n_heads * head_size)
            TensorView<value_type> wk = m_weights.wk.slice(layer_idx);  // (dim, n_kv_heads * head_size)
            TensorView<value_type> wv = m_weights.wv.slice(layer_idx);  // (dim, n_kv_heads * head_size)
            auto attention = std::make_unique<Attention<COMPUTE, value_type>>(wq, wk, wv, kv_dim, dim, n_heads, n_kv_heads, seq_len);

            // FF layer
            TensorView<value_type> w1 = m_weights.w1.slice(layer_idx);  //(hidden_dim, dim)
            TensorView<value_type> w2 = m_weights.w2.slice(layer_idx);  //(dim,hidden_dim)
            TensorView<value_type> w3 = m_weights.w3.slice(layer_idx);  //(hidden_dim, dim)
            auto feedforward = std::make_unique<FeedForward<COMPUTE, value_type>>(w1, w2, w3, dim, hidden_dim);

            // TensorBlock
            TensorView<value_type> wo = m_weights.wo.slice(layer_idx);                     // (dim, dim)
            TensorView<value_type> w_rms_att = m_weights.rms_att_weight.slice(layer_idx);  // (dim)
            TensorView<value_type> w_rms_ffn = m_weights.rms_ffn_weight.slice(layer_idx);  // (dim)
            auto tensorblock =
                std::make_unique<TransformerBlock<COMPUTE, value_type>>(std::move(attention), std::move(feedforward), w_rms_ffn, wo, w_rms_att, dim);

            m_layers.push_back(std::move(tensorblock));
        }

        m_linear = std::make_unique<Linear<COMPUTE, value_type>>(m_weights.wcls);
        m_out_logits.reShape(Shape(m_linear->outDim()));
        m_x_in.reShape(Shape(dim));
    }

    ~Transformer() {}

    void forward(int token, int pos, Tensor<CPU, value_type> &logits) {
        // copy the token embedding into x
        TensorView<value_type> content_row = m_weights.token_embedding_table.slice(token);  // (dim)

        // Prepare input tensors. Copy from CPU to device
        m_x_in.copyFrom(content_row);

        // forward all the layers
        for (auto &layer : m_layers) {
            layer->forward(m_x_in, pos);
        }

        // final rmsnorm
        rmsnorm(m_x_in, m_x_in, m_weights.rms_final_weight);

        // classifier into logits
        m_linear->forward(m_x_in, m_out_logits);

        // Copy from tensor on COMPUTE to a CPU Tensor.
        logits.copyFrom(m_out_logits);
    }

    auto getConfig() const -> const TransformerConfig & { return m_config; }

   private:
    TransformerConfig m_config;                                                 // the hyperparameters of the architecture (the blueprint)
    TransformerWeights<COMPUTE, value_type> m_weights;                          // the weights of the model
    Linear<COMPUTE, value_type>::ptr m_linear;                                  // linear layer
    Tensor<COMPUTE, value_type> m_x_in;                                         // input tensor.
    Tensor<COMPUTE, value_type> m_out_logits;                                   // logits output buffer on COMPUTE
    std::vector<typename TransformerBlock<COMPUTE, value_type>::ptr> m_layers;  // transformer block layers
};

}  // namespace llama2cpp
#endif