#ifndef ak_rmsnorm_H
#define ak_rmsnorm_H

// RMSNorm (root mean square normalization) divides each element in weight
// by the root mean square error of all elements in x and stores the result in o

static inline
float ak_calculate_sum_of_squares(float* x, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
         ss += x[j] * x[j];
     }
     return ss;
}

static inline
void ak_rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = ak_calculate_sum_of_squares(x, size);
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]); // same as (weight[j] * x[j])/sqrtf(ss);
    }
}

#endif