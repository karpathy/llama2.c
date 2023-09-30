#ifndef _ak_sampler_H
#define _ak_sampler_H

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token index
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct ak_sampler_s;
typedef struct ak_sampler_s ak_sampler_t;

ak_sampler_t* ak_sampler_init(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void ak_sampler_destroy(ak_sampler_t* sampler);

int ak_sampler_sample(ak_sampler_t* sampler, float* logits);

#include "ak_sampler.c"

#endif