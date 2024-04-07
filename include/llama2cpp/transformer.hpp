#ifndef LLAMA2CPP_TRANSFORMER_HPP
#define LLAMA2CPP_TRANSFORMER_HPP
#include <string>
#include <cstdlib>
#include <memory>
#include <llama2cpp/ops.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <llama2cpp/tensor.hpp>

namespace llama2cpp
{

    /**
     * @brief Transformer configuration
     *
     */
    struct TransformerConfig
    {
        int dim;        // transformer dimension
        int hidden_dim; // for ffn layers
        int n_layers;   // number of layers
        int n_heads;    // number of query heads
        int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        int vocab_size; // vocabulary size, usually 256 (byte-level)
        int seq_len;    // max sequence length
    };

    struct TransformerWeights
    {
        // token embedding table
        float *token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        float *rms_att_weight; // (layer, dim) rmsnorm weights
        float *rms_ffn_weight; // (layer, dim)
        // weights for matmuls. note dim == n_heads * head_size
        float *wq; // (layer, dim, n_heads * head_size)
        float *wk; // (layer, dim, n_kv_heads * head_size)
        float *wv; // (layer, dim, n_kv_heads * head_size)
        float *wo; // (layer, n_heads * head_size, dim)
        // weights for ffn
        float *w1; // (layer, hidden_dim, dim)
        float *w2; // (layer, dim, hidden_dim)
        float *w3; // (layer, hidden_dim, dim)
        // final rmsnorm
        float *rms_final_weight; // (dim,)
        // (optional) classifier weights for the logits, on the last layer
        float *wcls;
    };

    struct RunState
    {
        // current wave of activations
        float *x;      // activation at current time stamp (dim,)
        float *xb;     // same, but inside a residual branch (dim,)
        float *xb2;    // an additional buffer just for convenience (dim,)
        float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
        float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
        float *q;      // query (dim,)
        float *k;      // key (dim,)
        float *v;      // value (dim,)
        float *att;    // buffer for scores/attention values (n_heads, seq_len)
        float *logits; // output logits
        // kv cache
        float *key_cache;   // (layer, seq_len, dim)
        float *value_cache; // (layer, seq_len, dim)
    };

    void memory_map_weights(TransformerWeights *weights, TransformerConfig &config, float *ptr, int shared_weights)
    {
        int head_size = config.dim / config.n_heads;
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        unsigned long long n_layers = config.n_layers;
        weights->token_embedding_table = ptr;
        ptr += config.vocab_size * config.dim;
        weights->rms_att_weight = ptr;
        ptr += n_layers * config.dim;
        weights->wq = ptr;
        ptr += n_layers * config.dim * (config.n_heads * head_size);
        weights->wk = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights->wv = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights->wo = ptr;
        ptr += n_layers * (config.n_heads * head_size) * config.dim;
        weights->rms_ffn_weight = ptr;
        ptr += n_layers * config.dim;
        weights->w1 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights->w2 = ptr;
        ptr += n_layers * config.hidden_dim * config.dim;
        weights->w3 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights->rms_final_weight = ptr;
        ptr += config.dim;
        ptr += config.seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += config.seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        weights->wcls = shared_weights ? weights->token_embedding_table : ptr;
    }

    void read_checkpoint(const std::string &checkpoint_path, TransformerConfig &config, TransformerWeights *weights,
                         int *fd, float **data, ssize_t *file_size)
    {
        FILE *file = fopen(checkpoint_path.c_str(), "rb");
        if (!file)
        {
            fprintf(stderr, "Couldn't open file %s\n", checkpoint_path.c_str());
            exit(EXIT_FAILURE);
        }
        // read in the config header
        if (fread(&config, sizeof(TransformerConfig), 1, file) != 1)
        {
            exit(EXIT_FAILURE);
        }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        *fd = open(checkpoint_path.c_str(), O_RDONLY); // open in read only mode
        if (*fd == -1)
        {
            fprintf(stderr, "open failed!\n");
            exit(EXIT_FAILURE);
        }
        *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED)
        {
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
        // // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        // int shared_weights = config->vocab_size > 0 ? 1 : 0;
        // config->vocab_size = abs(config->vocab_size);
        // // figure out the file size
    }

    void malloc_run_state(RunState &s, TransformerConfig &config)
    {
        // we calloc instead of malloc to keep valgrind happy
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        s.x = (float *)calloc(config.dim, sizeof(float));
        s.xb = (float *)calloc(config.dim, sizeof(float));
        s.xb2 = (float *)calloc(config.dim, sizeof(float));
        s.hb = (float *)calloc(config.hidden_dim, sizeof(float));
        s.hb2 = (float *)calloc(config.hidden_dim, sizeof(float));
        s.q = (float *)calloc(config.dim, sizeof(float));
        s.key_cache = (float *)calloc(config.n_layers * config.seq_len * kv_dim, sizeof(float));
        s.value_cache = (float *)calloc(config.n_layers * config.seq_len * kv_dim, sizeof(float));
        s.att = (float *)calloc(config.n_heads * config.seq_len, sizeof(float));
        s.logits = (float *)calloc(config.vocab_size, sizeof(float));
        // ensure all mallocs went fine
        if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 || !s.q || !s.key_cache || !s.value_cache || !s.att || !s.logits)
        {
            std::cerr << "malloc failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void free_run_state(RunState &s)
    {
        free(s.x);
        free(s.xb);
        free(s.xb2);
        free(s.hb);
        free(s.hb2);
        free(s.q);
        free(s.att);
        free(s.logits);
        free(s.key_cache);
        free(s.value_cache);
    }

    class Attention
    {
    public:
        using ptr = std::unique_ptr<Attention>;
        Attention(float *wq, float *wk, float *wv, float *key_cache, float *value_cache, int loff, int kv_dim, int dim, int n_heads, int kv_heads, int seq_len)
            : m_wq(wq), m_wk(wk), m_wv(wv), m_key_cache(key_cache), m_value_cache(value_cache), m_loff(loff), m_kv_dim(kv_dim), m_dim(dim), m_n_heads(n_heads), m_kv_heads(kv_heads), m_head_size(dim / n_heads), m_seq_len(seq_len) {}

        void forward(float *in, float *q, float *k, float *v, float *att, float *xb, int pos_)
        {
            // key and value point to the kv cache
            k = m_key_cache + m_loff + pos_ * m_kv_dim;
            v = m_value_cache + m_loff + pos_ * m_kv_dim;

            // qkv matmuls for this position
            matmul(q, in, m_wq, m_dim, m_dim);
            matmul(k, in, m_wk, m_dim, m_kv_dim);
            matmul(v, in, m_wv, m_dim, m_kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < m_dim; i += 2)
            {
                int head_dim = i % m_head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)m_head_size);
                float val = pos_ * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < m_kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++)
                {
                    float *vec = v == 0 ? q : k; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // multihead attention. iterate over all heads
            int kv_mul = m_n_heads / m_kv_heads;
            int h;
#pragma omp parallel for private(h)
            for (h = 0; h < m_n_heads; h++)
            {
                // get the query vector for this head
                float *q_ = q + h * m_head_size;
                // attention scores for this head
                float *att_ = att + h * m_seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos_; t++)
                {
                    // get the key vector for this head and at this timestep
                    float *k_ = m_key_cache + m_loff + t * m_kv_dim + (h / kv_mul) * m_head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < m_head_size; i++)
                    {
                        score += q_[i] * k_[i];
                    }
                    score /= sqrtf(m_head_size);
                    // save the score to the attention buffer
                    att_[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att_, pos_ + 1);

                // weighted sum of the values, store back into xb
                float *xb_ = xb + h * m_head_size;
                memset(xb_, 0, m_head_size * sizeof(float));
                for (int t = 0; t <= pos_; t++)
                {
                    // get the value vector for this head and at this timestep
                    float *v_ = m_value_cache + m_loff + t * m_kv_dim + (h / kv_mul) * m_head_size;
                    // get the attention weight for this timestep
                    float a = att_[t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < m_head_size; i++)
                    {
                        xb_[i] += a * v_[i];
                    }
                }
            }
        }

    private:
        float *m_wq;
        float *m_wk;
        float *m_wv;
        float *m_key_cache;
        float *m_value_cache;
        int m_loff; // kv cache layer offset for convenience
        int m_kv_dim;
        int m_dim;
        int m_n_heads;
        int m_kv_heads;
        int m_head_size;
        int m_seq_len;
    };

    class FeedForward
    {
    public:
        using ptr = std::unique_ptr<FeedForward>;
        FeedForward(float *w1_, float *w2_, float *w3_, float *hb_, float *hb2_, int dim, int hidden_dim)
            : m_dim(dim), m_hidden_dim(hidden_dim), m_w1(w1_), m_w2(w2_), m_w3(w3_), m_hb(hb_), m_hb2(hb2_) {}

        /**
         * @brief forward pass for feedforward layer.
         *
         * self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
         */
        void forward(float *in, float *out)
        {
            matmul(m_hb, in, m_w1, m_dim, m_hidden_dim);
            matmul(m_hb2, in, m_w3, m_dim, m_hidden_dim);

            // SwiGLU non-linearity
            for (int i = 0; i < m_hidden_dim; i++)
            {
                float val = m_hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0f / (1.0f + expf(-val)));
                // elementwise multiply with w3(x)
                val *= m_hb2[i];
                m_hb[i] = val;
            }

            // final matmul to get the output of the ffn
            matmul(out, m_hb, m_w2, m_hidden_dim, m_dim);
        }

        auto dim() const -> const int { return m_dim; }

    private:
        int m_dim; // transformer dimension.
        int m_hidden_dim;
        float *m_w1;  // (layer, hidden_dim, dim)
        float *m_w2;  // (layer, hidden_dim, dim)
        float *m_w3;  // (layer, hidden_dim, dim)
        float *m_hb;  // buffer for hidden dimension (hidden_dim,)
        float *m_hb2; // buffer for hidden dimension (hidden_dim,)
    };

    class TransformerBlock
    {
    public:
        using ptr = std::unique_ptr<TransformerBlock>;
        TransformerBlock(FeedForward::ptr feed_forward, float *rms_ffn_weight)
            : m_attention(nullptr), m_feedforward(std::move(feed_forward)), m_rms_ffn_weight(rms_ffn_weight)
        {
        }

        void forward(float *x, float *h, float *out)
        {
            // attention norm

            // forward attention.

            // ffn rmsnorm
            rmsnorm(h, x, m_rms_ffn_weight, m_feedforward->dim());
            // forward FFN.
            m_feedforward->forward(h, out);

            // residual connection
            for (int i = 0; i < m_feedforward->dim(); i++)
            {
                x[i] += out[i];
            }
        }

    private:
        Attention::ptr m_attention;
        FeedForward::ptr m_feedforward;
        float *m_rms_ffn_weight;
    };

    class Linear
    {
    public:
        using ptr = std::unique_ptr<Linear>;

        Linear(float *wcls, int in_dim, int out_dim) : m_wcls(wcls), m_in_dim(in_dim), m_out_dim(out_dim) {}

        void forward(float *x, float *out)
        {
            matmul(out, x, m_wcls, m_in_dim, m_out_dim);
        }

    private:
        float *m_wcls;
        int m_in_dim;
        int m_out_dim;
    };

    class Transformer
    {
    public:
        using ptr = std::unique_ptr<Transformer>;
        Transformer(const std::string &checkpoint_path)
        {
            // read in the Config and the Weights from the checkpoint
            read_checkpoint(checkpoint_path, m_config, &m_weights, &m_fd, &m_data, &m_file_size);
            // allocate the RunState buffers
            malloc_run_state(m_state, m_config);

            initializeLayers();
        }

        void initializeLayers()
        {
            int kv_dim = (m_config.dim * m_config.n_kv_heads) / m_config.n_heads;
            for (unsigned long long l = 0; l < m_config.n_layers; l++)
            {
                // Attention layer
                int loff_ = l * m_config.seq_len * kv_dim; // kv cache layer offset for convenience
                float *wq = m_weights.wq + l * m_config.dim * m_config.dim;
                float *wk = m_weights.wk + l * m_config.dim * kv_dim;
                float *wv = m_weights.wv + l * m_config.dim * kv_dim;
                int head_size = m_config.dim / m_config.n_heads;
                Attention::ptr attention = std::make_unique<Attention>(wq, wk, wv,
                                                                       m_state.key_cache,
                                                                       m_state.value_cache,
                                                                       loff_, kv_dim,
                                                                       m_config.dim,
                                                                       m_config.n_heads,
                                                                       m_config.n_kv_heads,
                                                                       m_config.seq_len);
                m_attn_layers.push_back(std::move(attention));

                // FF layer
                float *w1 = m_weights.w1 + l * m_config.dim * m_config.hidden_dim;
                float *w2 = m_weights.w2 + l * m_config.dim * m_config.hidden_dim;
                float *w3 = m_weights.w3 + l * m_config.dim * m_config.hidden_dim;
                float *rms_ffn_weight_ = m_weights.rms_ffn_weight + l * m_config.dim;
                FeedForward::ptr feedforward = std::make_unique<FeedForward>(w1, w2, w3,
                                                                             m_state.hb,
                                                                             m_state.hb2,
                                                                             m_config.dim,
                                                                             m_config.hidden_dim);
                m_layers.push_back(std::make_unique<TransformerBlock>(std::move(feedforward), rms_ffn_weight_));
            }

            m_linear = std::make_unique<Linear>(m_weights.wcls, m_config.dim, m_config.vocab_size);
        }

        ~Transformer()
        {
            // close the memory mapping
            if (m_data != MAP_FAILED)
            {
                munmap(m_data, m_file_size);
            }
            if (m_fd != -1)
            {
                close(m_fd);
            }
            // free the RunState buffers
            free_run_state(m_state);
        }

        auto forward(int token, int pos) -> float32_t *
        {
            // a few convenience variables
            TransformerWeights *w = &m_weights;
            RunState *s = &m_state;
            float *x = s->x;
            int dim = m_config.dim;
            int kv_dim = (m_config.dim * m_config.n_kv_heads) / m_config.n_heads;
            int kv_mul = m_config.n_heads / m_config.n_kv_heads; // integer multiplier of the kv sharing in multiquery
            int hidden_dim = m_config.hidden_dim;
            int head_size = dim / m_config.n_heads;

            // copy the token embedding into x
            float *content_row = w->token_embedding_table + token * dim;
            memcpy(x, content_row, dim * sizeof(*x));

            // forward all the layers
            for (unsigned long long l = 0; l < m_config.n_layers; l++)
            {

                // attention rmsnorm
                rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);


                m_attn_layers[l]->forward(s->xb, s->q, s->k, s->v, s->att, s->xb, pos);


                // final matmul to get the output of the attention
                matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

                // residual connection back into x
                for (int i = 0; i < dim; i++)
                {
                    x[i] += s->xb2[i];
                }

                m_layers[l]->forward(x, s->xb, s->xb);
            }

            // final rmsnorm
            rmsnorm(x, x, w->rms_final_weight, dim);

            // classifier into logits
            m_linear->forward(x, s->logits);
            return s->logits;
        }

        auto getConfig() const -> const TransformerConfig &
        {
            return m_config;
        }

    private:
        TransformerConfig m_config;   // the hyperparameters of the architecture (the blueprint)
        TransformerWeights m_weights; // the weights of the model
        RunState m_state;             // buffers for the "wave" of activations in the forward pass
        // some more state needed to properly clean up the memory mapping (sigh)
        int m_fd;            // file descriptor for memory mapping
        float *m_data;       // memory mapped data pointer
        ssize_t m_file_size; // size of the checkpoint file in bytes
        Linear::ptr m_linear;
        std::vector<TransformerBlock::ptr> m_layers;
        std::vector<Attention::ptr> m_attn_layers;
    };

}
#endif