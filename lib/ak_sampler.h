#ifndef _ak_sampler_H
#define _ak_sampler_H

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token index
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct ak_sampler_s;
typedef struct ak_sampler_s ak_sampler_t;

/*
   vocab_size - the size of the vocabulary

   temperature - In language models like GPT, "temperature" is a hyperparameter
      used during sampling to control the randomness of the model's output.
      A higher temperature (e.g., 1.0 or above) makes the output more random,
      while a lower temperature (e.g., 0.1) makes it more focused and deterministic.

      If the temperature is 0.0, then the token with the highest probability is chosen.

      ak_params chooses 1.0 which causes no effect and the comment sets to not set higher

   topp - Top-p sampling in language models is used to narrow down the pool of
      next-word choices to those that have a cumulative probability of at least
      'p'. It provides a balance between diversity and relevance in generated
      text by only considering the most probable tokens whose probabilities
      sum up to 'p' or more.

      If topp <= 0 or >= 1, the Top-p is not used and a random number is chosen
      between 0 and 1

      ak_params chooses 0.9 for top-p in nucleus sampling.  1.0 is off, 0.9 works well, but slower

   rng_seed - if zero, time will be the seed, otherwise this number will be the number given.
       ak_params defaults to zero (or time)
*/

ak_sampler_t* ak_sampler_init(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void ak_sampler_destroy(ak_sampler_t* sampler);

/*
    The logits are the result of the forward pass from ak_transformer.  The logits represent a
    classification which might be like

    { 0.01, 0.01, 0.2, 0.3, 0.25, ... } (sum = 1.0)

    0.3 (3) would be the most likely candidate, followed by 0.25 (4), followed by 0.2 (2)
*/
int ak_sampler_sample(ak_sampler_t* sampler, float* logits);

#include "ak_sampler.c"

#endif