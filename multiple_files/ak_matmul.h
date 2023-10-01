#ifndef _ak_matmul_H
#define _ak_matmul_H

/* If PTHREADS=8, then 8 pthreads will be used */
#ifndef PTHREADS

static
void ak_matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

#else

#include <pthread.h>

typedef struct {
    float *xout, *x, *w, *ew;
    int n;
    pthread_t thread;
} matmul_t;

void *matmul_thread(void *arg) {
    matmul_t *mm = (matmul_t *)arg;
    float *w = mm->w;
    float *xout = mm->xout;
    while(w < mm->ew) {
        float val = 0.0f;
        float *x = mm->x;
        float *nw = w+mm->n;
        while(w < nw) {
            val += *w * *x;
            w++;
            x++;
        }
        *xout++ = val;
    }
    return NULL;
}

static
void ak_matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    // #pragma omp parallel for private(i)
    const int num_threads = PTHREADS;
    int chunk_size = d / num_threads;
    int start = 0;
    matmul_t *mm = (matmul_t *)malloc(sizeof(*mm) * num_threads);
    for (i = 0; i < num_threads; i++) {
        int next = i+1 < num_threads ? start+chunk_size : d;
        mm[i].xout = xout + start;
        mm[i].x = x;
        mm[i].w = w + (start * n);
        mm[i].ew = w + (next * n);
        mm[i].n = n;
        pthread_create(&mm[i].thread, NULL, matmul_thread, mm+i );
        start += chunk_size;
    }
    for (i = 0; i < num_threads; i++) {
        pthread_join(mm[i].thread, NULL);
    }
    free(mm);
}

#endif /* PTHREADS */
#endif /* ak_matmul_H */