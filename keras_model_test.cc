
#include "keras_model.h"

#include <stdio.h>
#include <iostream>

namespace kerasify {
#include "test_benchmark.h"
#include "test_conv_2x2.h"
#include "test_conv_3x3.h"
#include "test_conv_3x3x3.h"
#include "test_conv_softplus_2x2.h"
#include "test_dense_10x1.h"
#include "test_dense_10x10.h"
#include "test_dense_10x10x10.h"
#include "test_dense_1x1.h"
#include "test_dense_2x2.h"
#include "test_dense_relu_10.h"
#include "test_elu_10.h"
#include "test_func_conv_2x2.h"
#include "test_func_conv_2x2_multi_in_out.h"
#include "test_func_dense_1x1.h"
#include "test_func_dense_1x1_multi_in.h"
#include "test_func_dense_1x1_multi_in_out.h"
#include "test_func_dense_1x1_multi_out.h"
#include "test_func_maxpool2d_3x3x3.h"
#include "test_func_merge_1x1.h"
#include "test_maxpool2d_1x1.h"
#include "test_maxpool2d_2x2.h"
#include "test_maxpool2d_3x2x2.h"
#include "test_maxpool2d_3x3x3.h"
#include "test_relu_10.h"

bool tensor_test() {
  {
    const int i = 3;
    const int j = 5;
    const int k = 10;
    Tensor t(i, j, k);

    float c = 1.f;
    for (int ii = 0; ii < i; ii++) {
      for (int jj = 0; jj < j; jj++) {
        for (int kk = 0; kk < k; kk++) {
          t(ii, jj, kk) = c;
          c += 1.f;
        }
      }
    }

    c = 1.f;
    int cc = 0;
    for (int ii = 0; ii < i; ii++) {
      for (int jj = 0; jj < j; jj++) {
        for (int kk = 0; kk < k; kk++) {
          KASSERT_EQ(t(ii, jj, kk), c, 1e-9);
          KASSERT_EQ(t.data_[cc], c, 1e-9);
          c += 1.f;
          cc++;
        }
      }
    }
  }

  {
    const int i = 2;
    const int j = 3;
    const int k = 4;
    const int l = 5;
    Tensor t(i, j, k, l);

    float c = 1.f;
    for (int ii = 0; ii < i; ii++) {
      for (int jj = 0; jj < j; jj++) {
        for (int kk = 0; kk < k; kk++) {
          for (int ll = 0; ll < l; ll++) {
            t(ii, jj, kk, ll) = c;
            c += 1.f;
          }
        }
      }
    }

    c = 1.f;
    int cc = 0;
    for (int ii = 0; ii < i; ii++) {
      for (int jj = 0; jj < j; jj++) {
        for (int kk = 0; kk < k; kk++) {
          for (int ll = 0; ll < l; ll++) {
            KASSERT_EQ(t(ii, jj, kk, ll), c, 1e-9);
            KASSERT_EQ(t.data_[cc], c, 1e-9);
            c += 1.f;
            cc++;
          }
        }
      }
    }
  }

  return true;
}

}  // namespace kerasify

int main() {
  double load_time = 0.0;
  double apply_time = 0.0;

  if (!kerasify::tensor_test()) return 1;

  if (!kerasify::test_conv_2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_dense_1x1(&load_time, &apply_time)) return 1;

  if (!kerasify::test_dense_10x1(&load_time, &apply_time)) return 1;

  if (!kerasify::test_dense_2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_dense_10x10(&load_time, &apply_time)) return 1;

  if (!kerasify::test_dense_10x10x10(&load_time, &apply_time)) return 1;

  if (!kerasify::test_conv_2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_conv_3x3(&load_time, &apply_time)) return 1;

  if (!kerasify::test_conv_3x3x3(&load_time, &apply_time)) return 1;

  if (!kerasify::test_elu_10(&load_time, &apply_time)) return 1;

  if (!kerasify::test_relu_10(&load_time, &apply_time)) return 1;

  if (!kerasify::test_dense_relu_10(&load_time, &apply_time)) return 1;

  if (!kerasify::test_conv_softplus_2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_maxpool2d_1x1(&load_time, &apply_time)) return 1;

  if (!kerasify::test_maxpool2d_2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_maxpool2d_3x2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_maxpool2d_3x3x3(&load_time, &apply_time)) return 1;

  if (!kerasify::test_func_dense_1x1(&load_time, &apply_time)) return 1;

  if (!kerasify::test_func_conv_2x2(&load_time, &apply_time)) return 1;

  if (!kerasify::test_func_maxpool2d_3x3x3(&load_time, &apply_time)) return 1;

  if (!kerasify::test_func_merge_1x1(&load_time, &apply_time)) return 1;

  if (!kerasify::test_func_dense_1x1_multi_in(&load_time, &apply_time))
    return 1;

  if (!kerasify::test_func_dense_1x1_multi_out(&load_time, &apply_time))
    return 1;

  if (!kerasify::test_func_dense_1x1_multi_in_out(&load_time,
                                                        &apply_time))
    return 1;

  if (!kerasify::test_func_conv_2x2_multi_in_out(&load_time, &apply_time))
    return 1;

  // Run benchmark 5 times and report duration.
  double total_load_time = 0.0;
  double total_apply_time = 0.0;

  for (int i = 0; i < 5; i++) {
    if (!kerasify::test_benchmark(&load_time, &apply_time)) return 1;

    total_load_time += load_time;
    total_apply_time += apply_time;
  }

  printf("Benchmark network loads in %fs\n", total_load_time / 5);
  printf("Benchmark network runs in %fs\n", total_apply_time / 5);

  return 0;
}
