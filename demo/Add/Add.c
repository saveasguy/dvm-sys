#define N (1024 * 32)

#pragma dvm array distribute[block][block]
int A[N][N];
#pragma dvm array align([i][j] with A[i][j])
int B[N][N];

int main() {
    #pragma dvm parallel ([i][j] on A[i][j])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = i;
            B[i][j] = i;
        }
    }
    dvmh_barrier();
    double startt = dvmh_wtime();
    #pragma dvm region inout(B) inout(A)
    {
        #pragma dvm parallel ([i][j] on A[i][j])
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                A[i][j] = A[i][j] + B[i][j];
            }
    }
    dvmh_barrier();
    double endt = dvmh_wtime();
    printf("Time in seconds = %2.4lf\n", endt - startt);
}
