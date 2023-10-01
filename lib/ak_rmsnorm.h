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

/*
    RMS redistributes numbers within an array such that the mean of the sum of the
    squares is 1.0

    When RMS is applied, it takes the sum of the squares

    [ 0.3^2, 0.6^2, 0.9^2 ] => 0.09 + 0.36 + 0.81 => 1.26

    Divides that by the number of items in the array (3)

    1.26 / 3 => 0.42

    Adds a small amount to it

    0.42 + 1e-5f

    Then divides each number by the square root of that

    [
        0.3 / sqrt(0.42 + 1e-5f),
        0.6 / sqrt(0.42 + 1e-5f),
        0.9 / sqrt(0.42 + 1e-5f)
    ] =>

    sqrt(0.42 + 1e-5f) = 0.6481

    [ 0.3 / 0.6481, 0.6 / 0.6481, 0.9 / 0.6481 ] =>

    [ 0.4629, 0.9258, 1.3887 ]

    The normalization is such that if we again compute the average
    of the sum of the squares, it will be 1.

    [ 0.4629^2, 0.9258^2, 1.3887^2 ] => 0.2143 + 0.8571 + 1.9285 => 3

    3 / 3 => 1
*/

static inline
void ak_rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = ak_calculate_sum_of_squares(x, size);
    // calculate the mean or average of the sum of the squares
    ss /= size;
    /* numerical stability is important to prevent issues like overflow or underflow
       in floating-point arithmetic. This small constant is added to the sum of squares
       divided by the size to ensure that the value inside the square root isn't zero.
       This avoids division by zero and potential underflow when taking the reciprocal
       of the square root. Numerical stability techniques are crucial for maintaining
       the accuracy and reliability of computations, especially in iterative methods
       like neural network training.
    */
    ss += 1e-5f;
    // take the reciprocal of the square root which should not have a divide by
    // zero due to line above.
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        // ss * x[j] == normalized
        // weight[j] * ^^^ = scaled
        o[j] = weight[j] * (ss * x[j]); // same as (weight[j] * x[j])/sqrtf(ss);
    }
}

#endif