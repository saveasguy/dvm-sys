#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pronem.h"

#define NET 512
double A[NET][NET];
double V[NET - 1][NET - 1];
double F[NET - 1][NET - 1];
double W[NET - 1];
double U[NET - 1][NET - 1];
int main(int argc, char *argv[]) {
  int NTIME = 10, N = NET - 1, I, J, KTAU;
  double H = 1. / NET, HH = H * H, R = 0.025, TAU = R * HH;
#pragma dvm actual(F, H, HH, I, N, V)
#pragma dvm region in(F, H, HH, I, N, V)out(F, I, V) local(J)
  {
#pragma dvm parallel([I]) tie(F[I][], V[I][]) private(J)
    for (I = 1; I < N; I++) {
      double X = I * H;
      double S = X * (1 - X);
      for (J = 1; J < N; J++) {
        double Y1 = (J - 1) * H, Y2 = (J + 1) * H,
               Q = 0.5 * (Y1 * (1 - Y1) + Y2 * (1 - Y2) + 1.67 * HH);
        V[I][J] = 96 * S * Q;
        F[I][J] = 192 * HH * (S + Q);
      }
    }
  }
#pragma dvm get_actual(F, I, V)

  for (I = 0; I < N + 1; I++) {
    A[I][0] = 0.;
    A[I][N] = 0.;
  }
#pragma dvm actual(A, J, N)
#pragma dvm region in(A, J, N)out(A, J)
  {
#pragma dvm parallel([J]) tie(A[][J])
    for (J = 1; J < N; J++) {
      A[0][J] = 0.;
      A[N][J] = 0.;
    }
  }
#pragma dvm get_actual(A, J)

  for (I = 1; I < N; I++) {
#pragma dvm actual(I, J, N, V, W)
#pragma dvm region in(I, J, N, V, W)out(J, V, W)
    {
#pragma dvm parallel([J]) tie(V[][J], W[J])
      for (J = 1; J < N; J++) {
        W[J] = V[I][J];
      }
    }
#pragma dvm get_actual(J, V, W)

    PRONEM(N, W);
#pragma dvm actual(A, I, J, N, W)
#pragma dvm region in(A, I, J, N, W)out(A, J, W)
    {
#pragma dvm parallel([J]) tie(A[][J], W[J])
      for (J = 1; J <= N; J++)
        A[I][J] = W[J];
    }
#pragma dvm get_actual(A, J, W)
  }
  for (KTAU = 1; KTAU < NTIME; KTAU++) {
#pragma dvm actual(A, F, HH, I, N, R, V)
#pragma dvm region in(A, F, HH, I, N, R, V)out(A, F, I, V) local(J)
    {
#pragma dvm parallel([I][J]) tie(A[I][J], F[I][J], V[I][J])
      for (I = 1; I < N; I++) {
        for (J = 1; J < N; J++) {
          double A1 = A[I - 1][J - 1] + A[I - 1][J + 1] + A[I + 1][J - 1] +
                      A[I + 1][J + 1],
                 A2 = A[I - 1][J] + A[I][J - 1] + A[I + 1][J] + A[I][J + 1],
                 G = A1 + 4 * A2 - 20 * A[I][J] + 6 * HH * F[I][J], W1 = R * G,
                 W2 = R * (G + 0.5 * W1 * HH), W3 = R * (G + 0.5 * W2 * HH),
                 W4 = R * (G + W3 * HH);
          V[I][J] = V[I][J] + 0.167 * (W1 + 2 * W2 + 2 * W3 + W4);
        }
      }
    }
#pragma dvm get_actual(A, F, I, V)

    for (I = 1; I < N; I++) {
#pragma dvm actual(I, J, N, V, W)
#pragma dvm region in(I, J, N, V, W)out(J, V, W)
      {
#pragma dvm parallel([J]) tie(V[][J], W[J])
        for (J = 1; J < N; J++) {
          W[J] = V[I][J];
        }
      }
#pragma dvm get_actual(J, V, W)

      PRONEM(N, W);
#pragma dvm actual(A, I, J, N, W)
#pragma dvm region in(A, I, J, N, W)out(A, J, W)
      {
#pragma dvm parallel([J]) tie(A[][J], W[J])
        for (J = 1; J < N; J++) {
          A[I][J] = W[J];
        }
      }
#pragma dvm get_actual(A, J, W)
    }
#pragma dvm actual(A, I, N, U)
#pragma dvm region in(A, I, N, U)out(A, I, U) local(J)
    {
#pragma dvm parallel([I][J]) tie(A[I][J], U[I][J])
      for (I = 1; I < N; I++) {
        for (J = 1; J <= N; J++) {
          U[I][J] = A[I][J];
        }
      }
    }
#pragma dvm get_actual(A, I, U)
  }
  printf("%5d %5d\n", NET, NTIME);

  /*	for(I=1;I<=N;I++)
          {
                  for(J=1;J<=N;J++)
                          printf("%f ",U[I][J]);
                  printf("\n");
          }*/
  return (0);
}

void PRONEM(int N, double *F) {
  double *ALFA, *BETA, *A, *B, *C, AA, BB, F1, F2;
  int *M, *L, I, J;
  ALFA = (double *)malloc((N + 1) * sizeof(double));
  BETA = (double *)malloc((N + 1) * sizeof(double));
  A = (double *)malloc((N + 1) * sizeof(double));
  B = (double *)malloc((N + 1) * sizeof(double));
  C = (double *)malloc((N + 1) * sizeof(double));
  M = (int *)malloc((N + 1) * sizeof(int));
  L = (int *)malloc((N + 1) * sizeof(int));
  for (I = 1; I <= N; I++) {
    A[I] = 1;
    B[I] = 4;
    C[I] = 1;
  }
  AA = A[2];
  BB = B[1];
  F1 = F[1];
  F2 = F[2];
  L[1] = 1;
  for (I = 1; I <= N - 1; I++) {
    if (fabs(BB) > fabs(C[I])) {
      ALFA[I + 1] = -C[I] / BB;
      BETA[I + 1] = F1 / BB;
      BB = B[I + 1] + AA * ALFA[I + 1];
      F1 = F2 - AA * BETA[I + 1];
      AA = A[I + 2];
      F2 = F[I + 2];
      M[I + 1] = L[I];
      L[I + 1] = I + 1;
    } else {
      ALFA[I + 1] = -BB / C[I];
      BETA[I + 1] = F1 / C[I];
      BB = AA + B[I + 1] * ALFA[I + 1];
      F1 = F2 - B[I + 1] * BETA[I + 1];
      AA = A[I + 2] * ALFA[I + 1];
      F2 = F[I + 2] - A[I + 2] * BETA[I + 1];
      M[I + 1] = I + 1;
      L[I + 1] = L[I];
    }
  }
  F[L[N]] = F1 / BB;
  for (I = N - 1; I <= 1; I--)
    F[M[I + 1]] = ALFA[I + 1] * F[L[I + 1]] + BETA[I + 1];
  free(ALFA);
  free(BETA);
  free(A);
  free(B);
  free(C);
  free(M);
  free(L);
  return;
}
