#ifndef LLAMA2CPP_SAMPLER_HPP
#define LLAMA2CPP_SAMPLER_HPP
#include <string>
#include <cstdlib>
#include <memory>
#include <llama2cpp/ops.hpp>
#include <llama2cpp/types.hpp>

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

namespace llama2cpp
{
    /**
     * @brief struct used when sorting probabilities during top-p sampling
     *
     */
    struct ProbIndex
    {
        float32_t prob;
        int index;
    };

    int compare(const void *a, const void *b)
    {
        ProbIndex *a_ = (ProbIndex *)a;
        ProbIndex *b_ = (ProbIndex *)b;
        if (a_->prob > b_->prob)
            return -1;
        if (a_->prob < b_->prob)
            return 1;
        return 0;
    }

    class Sampler
    {
    public:
        using ptr = std::unique_ptr<Sampler>;

        Sampler(int vocab_size_, float32_t temperature_, float32_t topp_, unsigned long long rng_seed)
        {
            vocab_size = vocab_size_;
            temperature = temperature_;
            topp = topp_;
            rng_state = rng_seed;
            // buffer only used with nucleus sampling; may not need but it's ~small
            probindex = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex));
        }

        ~Sampler()
        {
            free(probindex);
        }

        auto sample(float32_t *logits) -> int
        {
            // sample the token given the logits and some hyperparameters
            int next;
            if (temperature == 0.0f)
            {
                // greedy argmax sampling: take the token with the highest probability
                next = sample_argmax(logits, vocab_size);
            }
            else
            {
                // apply the temperature to the logits
                for (int q = 0; q < vocab_size; q++)
                {
                    logits[q] /= temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                softmax(logits, vocab_size);
                // flip a (float) coin (this is our source of entropy for sampling)
                // @TODO: implement an entropy generator.
                float32_t coin = random_f32(&rng_state);
                // we sample from this distribution to get the next token
                if (topp <= 0 || topp >= 1)
                {
                    // simply sample from the predicted probability distribution
                    next = sample_mult(logits, vocab_size, coin);
                }
                else
                {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(logits, vocab_size, topp, probindex, coin);
                }
            }
            return next;
        }

    private:
        int sample_argmax(float32_t *probabilities, int n)
        {
            // return the index that has the highest probability
            int max_i = 0;
            float32_t max_p = probabilities[0];
            for (int i = 1; i < n; i++)
            {
                if (probabilities[i] > max_p)
                {
                    max_i = i;
                    max_p = probabilities[i];
                }
            }
            return max_i;
        }
        unsigned int random_u32(unsigned long long *state)
        {
            // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            return (*state * 0x2545F4914F6CDD1Dull) >> 32;
        }
        float random_f32(unsigned long long *state)
        { // random float32 in [0,1)
            return (random_u32(state) >> 8) / 16777216.0f;
        }

        int sample_mult(float32_t *probabilities, int n, float32_t coin)
        {
            // sample index from probabilities (they must sum to 1!)
            // coin is a random number in [0, 1), usually from random_f32()
            float32_t cdf = 0.0f;
            for (int i = 0; i < n; i++)
            {
                cdf += probabilities[i];
                if (coin < cdf)
                {
                    return i;
                }
            }
            return n - 1; // in case of rounding errors
        }

        int sample_topp(float32_t *probabilities, int n, float32_t topp, ProbIndex *probindex, float32_t coin)
        {
            // top-p sampling (or "nucleus sampling") samples from the smallest set of
            // tokens that exceed probability topp. This way we never sample tokens that
            // have very low probabilities and are less likely to go "off the rails".
            // coin is a random number in [0, 1), usually from random_f32()

            int n0 = 0;
            // quicksort indices in descending order of probabilities
            // values smaller than (1 - topp) / (n - 1) cannot be part of the result
            // so for efficiency we crop these out as candidates before sorting
            const float32_t cutoff = (1.0f - topp) / (n - 1);
            for (int i = 0; i < n; i++)
            {
                if (probabilities[i] >= cutoff)
                {
                    probindex[n0].index = i;
                    probindex[n0].prob = probabilities[i];
                    n0++;
                }
            }
            qsort(probindex, n0, sizeof(ProbIndex), compare);

            // truncate the list where cumulative probability exceeds topp
            float32_t cumulative_prob = 0.0f;
            int last_idx = n0 - 1; // in case of rounding errors consider all elements
            for (int i = 0; i < n0; i++)
            {
                cumulative_prob += probindex[i].prob;
                if (cumulative_prob > topp)
                {
                    last_idx = i;
                    break; // we've exceeded topp by including last_idx
                }
            }

            // sample from the truncated list
            float32_t r = coin * cumulative_prob;
            float32_t cdf = 0.0f;
            for (int i = 0; i <= last_idx; i++)
            {
                cdf += probindex[i].prob;
                if (r < cdf)
                {
                    return probindex[i].index;
                }
            }
            return probindex[last_idx].index; // in case of rounding errors
        }

        int vocab_size;
        ProbIndex *probindex; // buffer used in top-p sampling
        float32_t temperature;
        float32_t topp;
        unsigned long long rng_state;
    };

}
#endif