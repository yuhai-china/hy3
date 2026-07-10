#ifndef HY3_GPU_H
#define HY3_GPU_H

#include "hy3.h"

int hy3_gpu_init(hy3_model *m, int n_gpu_layers);
void hy3_gpu_free(hy3_model *m);
int hy3_gpu_eval(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos);

#endif
