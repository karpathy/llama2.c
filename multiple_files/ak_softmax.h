#ifndef _ak_softmax_H
#define _ak_softmax_H

//  Softmax exponentiates its input and normalizes it to create a probability distribution
//  where the sum of the elements is 1.  This alters the array x inline.

/*
    From ChatGPT

    If max_val wasn't used for numerical stability in the softmax function, you risk numerical
    instability issues:

        Overflow: When the exponential function expf(x[i]) is applied to a large value in x[i],
            it could result in a value too large to be represented, causing overflow.
        Underflow: When summing up the exponential terms, if the original values in x are
            significantly different from each other, smaller exponentials might be rounded
            down to zero, leading to a loss of information.

    By subtracting max_val from each element before taking the exponential, both overflow and underflow risks are mitigated, and the output remains mathematically equivalent due to the properties of the softmax function.
*/

static inline
void ak_softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

#endif