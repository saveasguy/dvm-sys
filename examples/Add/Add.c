#define N (1024)

#pragma dvm array distribute[block]
int A[N];
#pragma dvm array align([i] with A[i])
int B[N];

int main()
{
#pragma dvm parallel([i] on A[i])
    for (int i = 0; i < N; ++i)
    {
        A[i] = i;
        B[i] = i;
    }
    dvmh_barrier();
    double startt = dvmh_wtime();
#pragma dvm region inout(B) inout(A)
    {
#pragma dvm parallel([i] on A[i])
        for (int i = 0; i < N; ++i)
            A[i] = A[i] + B[i];
    }
    dvmh_barrier();
    double endt = dvmh_wtime();
    printf("Time in seconds = %2.4lf\n", endt - startt);
}
