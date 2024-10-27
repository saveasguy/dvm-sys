#define N (256)

#pragma dvm array distribute[block]
int A[N];
#pragma dvm array align([i] with A[i])
int B[N];

int main() {
    #pragma dvm parallel ([i] on A[i])
    for (int i = 0; i < N; ++i)
        A[i] += B[i];
}
