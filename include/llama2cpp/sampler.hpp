#ifndef LLAMA2CPP_SAMPLER_HPP
#define LLAMA2CPP_SAMPLER_HPP
#include <algorithm>
#include <cstdlib>
#include <llama2cpp/transformer/ops.hpp>
#include <llama2cpp/transformer/types.hpp>
#include <memory>
#include <string>

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

namespace llama2cpp {
/**
 * @brief struct used when sorting probabilities during top-p sampling
 *
 */
struct ProbIndex {
    float32_t prob;
    int index;
};

class Sampler {
   public:
    using ptr = std::unique_ptr<Sampler>;

    Sampler(int vocab_size_, float32_t temperature_, float32_t topp_, unsigned long long rng_seed)
        : m_vocab_size(vocab_size_), m_temperature(temperature_), m_topp(topp_), m_rng_state(rng_seed) {
        // buffer only used with nucleus sampling; may not need but it's ~small
        probindex.resize(m_vocab_size);
    }

    ~Sampler() {}

    auto sample(Tensor<CPU, float32_t> &logits) -> int {
        // sample the token given the logits and some hyperparameters
        int next;
        if (m_temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            next = argmax(logits);
        } else {
            // apply the temperature to the logits
            for (int q = 0; q < logits.size(); q++) {
                logits[q] /= m_temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(logits);
            // flip a (float) coin (this is our source of entropy for sampling)
            // @TODO: implement an entropy generator.
            float32_t coin = random_f32(&m_rng_state);
            // we sample from this distribution to get the next token
            if (m_topp <= 0 || m_topp >= 1) {
                // simply sample from the predicted probability distribution
                next = sample_mult(logits, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sample_topp(logits, m_topp, probindex, coin);
            }
        }
        return next;
    }

   private:
    unsigned int random_u32(unsigned long long *state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (*state * 0x2545F4914F6CDD1Dull) >> 32;
    }
    float32_t random_f32(unsigned long long *state) {  // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }

    int sample_mult(Tensor<CPU, float32_t> &probabilities, float32_t coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float32_t cdf = 0.0f;
        for (int i = 0; i < probabilities.size(); i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return probabilities.size() - 1;  // in case of rounding errors
    }

    int sample_topp(Tensor<CPU, float32_t> &probabilities, float32_t topp, std::vector<ProbIndex> &probindex, float32_t coin) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        int n0 = 0;
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        const float32_t cutoff = (1.0f - topp) / (probabilities.size() - 1);
        for (int i = 0; i < probabilities.size(); i++) {
            if (probabilities[i] >= cutoff) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }
        std::sort(probindex.data(), probindex.data() + n0, [](auto a, auto b) { return a.prob > b.prob; });

        // truncate the list where cumulative probability exceeds topp
        float32_t cumulative_prob = 0.0f;
        int last_idx = n0 - 1;  // in case of rounding errors consider all elements
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probindex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;  // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float32_t r = coin * cumulative_prob;
        float32_t cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probindex[i].prob;
            if (r < cdf) {
                return probindex[i].index;
            }
        }
        return probindex[last_idx].index;  // in case of rounding errors
    }

    int m_vocab_size;
    std::vector<ProbIndex> probindex;  // buffer used in top-p sampling
    float32_t m_temperature;
    float32_t m_topp;
    unsigned long long m_rng_state;
};

}  // namespace llama2cpp
#endif