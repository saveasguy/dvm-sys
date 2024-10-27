#include <math_constants.h>

__device__ __inline__ float pow(float x, char n)                   { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, short n)                  { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, long long int n)          { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, double n)                 { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, unsigned char n)          { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, unsigned short n)         { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, unsigned int n)           { return powf(x, (float)n); }
__device__ __inline__ float pow(float x, unsigned long long int n) { return powf(x, (float)n); }

__device__ __inline__ double pow(double x, char n)                   { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, short n)                  { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, float n)                  { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, long long int n)          { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, unsigned char n)          { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, unsigned short n)         { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, unsigned int n)           { return pow(x, (double)n); }
__device__ __inline__ double pow(double x, unsigned long long int n) { return pow(x, (double)n); }

__device__ __inline__ float pow(char x, float n)                   { return powf((float)x, n); }
__device__ __inline__ float pow(short x, float n)                  { return powf((float)x, n); }
__device__ __inline__ float pow(int x, float n)                    { return powf((float)x, n); }
__device__ __inline__ float pow(long long int x, float n)          { return powf((float)x, n); }
__device__ __inline__ float pow(unsigned char x, float n)          { return powf((float)x, n); }
__device__ __inline__ float pow(unsigned short x, float n)         { return powf((float)x, n); }
__device__ __inline__ float pow(unsigned int x, float n)           { return powf((float)x, n); }
__device__ __inline__ float pow(unsigned long long int x, float n) { return powf((float)x, n); }

__device__ __inline__ double pow(char x, double n)                   { return pow((double)x, n); }
__device__ __inline__ double pow(short x, double n)                  { return pow((double)x, n); }
__device__ __inline__ double pow(int x, double n)                    { return pow((double)x, n); }
__device__ __inline__ double pow(long long int x, double n)          { return pow((double)x, n); }
__device__ __inline__ double pow(unsigned char x, double n)          { return pow((double)x, n); }
__device__ __inline__ double pow(unsigned short x, double n)         { return pow((double)x, n); }
__device__ __inline__ double pow(unsigned int x, double n)           { return pow((double)x, n); }
__device__ __inline__ double pow(unsigned long long int x, double n) { return pow((double)x, n); }

template<typename T>
__device__ __inline__ int pow(int x, T n) {
    int p;
    if (x == 2)
        p = 1 << n;
    else {
        // XXX: What if both x and n are negative?
        if (n < 0)
            p = 0;// (int)powf(x, n);
        else if (n == 1)
            p = x;
        else {
            p = 1;
            int a = x;
            while (n > 0) {
                if ((n & 1) != 0)
                    p *= a;
                a *= a;
                n >>= 1;
            }
        }
    }
    return p;
}

// for windows with type long int
__device__ __inline__ long pow(long x, int n) {
    int p;
    if (x == 2)
        p = 1 << n;
    else {
        // XXX: What if both x and n are negative?
        if (n < 0)
            p = 0;// (int)powf(x, n);
        else if (n == 1)
            p = x;
        else {
            p = 1;
            int a = x;
            while (n > 0) {
                if ((n & 1) != 0)
                    p *= a;
                a *= a;
                n >>= 1;
            }
        }
    }
    return p;
}

__device__ __inline__ long long pow(long long x, int n) {
    long long p;
    if (x == 2)
        p = 1 << n;
    else {
        // XXX: What if both x and n are negative?
        if (n < 0)
            p = 0;// (int)powf(x, n);
        else if (n == 1)
            p = x;
        else {
            p = 1;
            long long a = x;
            while (n > 0) {
                if ((n & 1) != 0)
                    p *= a;
                a *= a;
                n >>= 1;
            }
        }
    }
    return p;
}

template<typename T>
__device__ __inline__ T shiftBits(T x, int pos) {
    T val = (sizeof(x) << 3) - 1;
    if (val > 31) val = 31;
    if (pos >= 0) {
        // XXX: C standard says that left-shift of signed integers involve undefined behavior unless it is non-negative and x*2^pos is representable
        //      Undefined behavior situations are more dangerous than implementation-defined ones.
        //      Fortran 95 standard says that if the absolute value of pos is greater than bit_size(x), the value is undefined.
        while (pos > val) {
            x <<= val;
            pos -= val;
        }
        return x << pos;
    } else {
        pos = -pos;
        // XXX: C standard leaves the right-shift of signed integers implementation-defined but Intel Fortran says precisely about the arithmetic shift
        //      Fortran 95 standard says that if the absolute value of pos is greater than bit_size(x), the value is undefined.
        while (pos > val) {
            x >>= val;
            pos -= val;
        }
        return x >> pos;
    }
}

template<typename T>
__device__ __inline__ T ibclr(T x, int pos) { return x & ~shiftBits<T>(1, pos); }

template<typename T>
__device__ __inline__ T ibchng(T x, int pos) { return x ^ shiftBits<T>(1, pos); }

template<typename T>
__device__ __inline__ T ibits(T x, int pos, int size) { return shiftBits<T>(x, -pos) & (shiftBits<T>(1, size) - 1); }

template<typename T>
__device__ __inline__ T ibset(T x, int pos) { return x | shiftBits<T>(1, pos); }

template<typename T>
__device__ __inline__ T ishft(T x, int shift) {
    if (shift >= 0) {
        return shiftBits<T>(x, shift);
    } else {
        shift = -shift - 1;
        x >>= 1;
        x = ibclr<T>(x, (sizeof(x) << 3) - 1);
        return shiftBits<T>(x, -shift);
    }
}

template<typename T>
__device__ __inline__ T lshft(T x, int shift) { return ishft<T>(x, shift); }

#ifndef INTEL_LOGICAL_TYPE
template<typename T>
__device__ __inline__ T rshft(T x, int shift) { return shiftBits<T>(x, -shift); }
#else
template<typename T>
__device__ __inline__ T rshft(T x, int shift) { return ishft<T>(x, -shift); }
#endif

template<typename T>
__device__ __inline__ T dshiftl(T l, T r, int shift) {
    if (shift == 0)
        return l;
    else if (shift == (sizeof(T) << 3))
        return r;
    else
        return ishft<T>(l, shift) | ishft<T>(r, shift - (sizeof(r) << 3));
}

template<typename T>
__device__ __inline__ T dshiftr(T l, T r, int shift) {
    if (shift == 0)
        return r;
    else if (shift == (sizeof(T) << 3))
        return l;
    else
        return ishft<T>(l, (sizeof(l) << 3) - shift) | ishft<T>(r, -shift);
}

template <typename T>
__device__ __inline__ T isha(T x, int shift) { return shiftBits<T>(x, shift); }

#ifndef INTEL_LOGICAL_TYPE
template <typename T>
__device__ __inline__ T shifta(T x, int shift) { return (shift == (sizeof(x) << 3) ? (x < 0 ? -1 : 0) : isha<T>(x, -shift)); }
#else
template <typename T>
__device__ __inline__ T shifta(T x, int shift) { return (shift == (sizeof(x) << 3) ? 0 : isha<T>(x, -shift)); }
#endif

#ifndef INTEL_LOGICAL_TYPE
template <typename T>
__device__ __inline__ T shiftr(T x, int shift) { return ishft<T>(x, -shift); }
#else
template <typename T>
__device__ __inline__ T shiftr(T x, int shift) { return rshft<T>(x, shift); }
#endif

template <typename T>
__device__ __inline__ T ishftc(T x, int shift, int size) {
    // XXX: Should it be so complex?
    T delim = shiftBits<T>(1, size - 1);
    T mask = delim | (delim - 1);
    T sub = x & mask;
    if (shift >= 0) {
        shift %= size;
        T sign = ((sub & delim) == delim);
        T count = shift;
        while (count != 0) {
            sub = ((sub << 1) & mask) | sign;
            sign = ((sub & delim) == delim);
            count--;
        }
    } else {
        shift = (-shift) % size;
        T sign = ((sub & 1) == 0 ? 0 : delim);
        T count = shift;
        while (count != 0) {
            sub = (((sub >> 1) & mask) & ~delim) | sign;
            sign = ((sub & 1) == 0 ? 0 : delim);
            count--;
        }
    }
    return sub | (x & ~mask);
}

template <typename T>
__device__ __inline__ T ishc(T x, int shift) { return ishftc<T>(x, shift, (sizeof(x) << 3)); }

template<typename T>
__device__ __inline__ T ilen(T x) {
    if (x < 0)
        x = -x - 1;
    T i = 0;
    while (x) {
        i++;
        x >>= 1;
    }
    return i;
}

template<typename T>
__device__ __inline__ T popcnt(T x) {
    T count;
    if (x >= 0) {
        count = 0;
    } else {
        count = 1;
        x = ibclr<T>(x, (sizeof(x) << 3) - 1);
    }
    while (x != 0) {
        if ((x & 1) != 0)
            count++;
        x >>= 1;
    }
    return count;
}

template<typename T>
__device__ __inline__ T trailz(T x) {
    T count = 0;
    while ((x & 1) == 0 && (count < (sizeof(x) << 3))) {
        count++;
        x >>= 1;
    }
    return count;
}

template<typename T>
__device__ __inline__ T copysign(T x, T y) { if (x < 0) x = -x; return ((y >= 0) ? x : -x); }

template<typename T>
__device__ __inline__ T fdim(T x, T y) { return ((x > y) ? (x - y) : 0); }

template<typename T>
__device__ __inline__ T fmod(T x, T y) { return x % y; }

template<typename T>
__device__ __inline__ T btest(T x, int pos) { return ((x & shiftBits<T>(1, pos)) != 0); }

template<typename T>
__device__ __inline__ float real(T x) { return (float)x; }

__device__ __inline__ double dprod(float x, float y) { return (double)x * (double)y; }

__device__ __inline__ char abs(char x) { return (char)abs((int)x); }
__device__ __inline__ short abs(short x) { return (short)abs((int)x); }

#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 < 7050)
    __device__ __inline__ long min(long x, long y) { return (long)min((long long)x, (long long)y); }
    __device__ __inline__ long max(long x, long y) { return (long)max((long long)x, (long long)y); }
#endif

__device__ __inline__ int min(int x, long long y) { return (int)min((long long)x, y); }
__device__ __inline__ int min(long long x, int y) { return (int)min(x, (long long)y); }
__device__ __inline__ int min(long x, int y) { return (int)min(x, (long )y); }
__device__ __inline__ int max(int x, long long y) { return (int)max((long long)x, y); }
__device__ __inline__ int max(long long x, int y) { return (int)max(x, (long long)y); }
__device__ __inline__ int max(long x, int y) { return (int)max(x, (long)y); }

template <typename T>
class Complex;

template <typename T>
inline __host__ __device__ T real(const Complex<T> &c1)
{ return c1.x; }

template <typename T>
inline __host__ __device__ T imag(const Complex<T> &c1)
{ return c1.y; }

template <typename T>
inline __host__ __device__ T abs(const Complex<T> &c1)
{ return hypot(c1.x, c1.y); }

template <typename T>
inline __host__ __device__ Complex<T> conj(const Complex<T> &c1)
{ return Complex<T>(c1.x, -c1.y); }

template <typename T>
inline __host__ __device__ Complex<T> sin(const Complex<T> &c1)
{ return Complex<T>(sin(c1.x) * cosh(c1.y), cos(c1.x) * sinh(c1.y)); }

template <typename T>
inline __host__ __device__ Complex<T> cos(const Complex<T> &c1)
{ return Complex<T>(cos(c1.x) * cosh(c1.y), -(sin(c1.x) * sinh(c1.y))); }

template <typename T>
inline __host__ __device__ Complex<T> tan(const Complex<T> &c1) {
    const T sin_x = sin(c1.x), cos_x = cos(c1.x);
    const T sinh_y = sinh(c1.y), cosh_y = cosh(c1.y);
    return Complex<T>(sin_x * cosh_y, cos_x * sinh_y) / Complex<T>(cos_x * cosh_y, -(sin_x * sinh_y));
}

template <typename T>
inline __host__ __device__ Complex<T> ctan(const Complex<T> &c1) {
    const T sin_x = sin(c1.x), cos_x = cos(c1.x);
    const T sinh_y = sinh(c1.y), cosh_y = cosh(c1.y);
    return Complex<T>(cos_x * cosh_y, -(sin_x * sinh_y)) / Complex<T>(sin_x * cosh_y, cos_x * sinh_y);
}

template <typename T>
inline __host__ __device__ Complex<T> log(const Complex<T> &c1)
{ return Complex<T>(log(abs(c1)), atan2(c1.y, c1.x)); }

template <typename T>
inline __host__ __device__ Complex<T> log10(const Complex<T> &c1)
{ return log(c1) / log(T(10)); }

template <typename T>
inline __host__ __device__ Complex<T> exp(const Complex<T> &c1) {
    const T exp_x = exp(c1.x);
    return Complex<T>(exp_x * cos(c1.y), exp_x * sin(c1.y));
}

template <typename T>
inline __host__ __device__ Complex<T> sqrt(const Complex<T> &c1) {
    T a, x, y;
    a = hypot(c1.x, c1.y);
    if (a == T(0)) {
        x = T(0);
        y = T(0);
    } else if (c1.x > T(0)) {
        x = sqrt(T(0.5) * (a + c1.x));
        y = T(0.5) * (c1.y / x);
    } else {
        y = sqrt(T(0.5) * (a - c1.x));
        if (c1.y < T(0))
            y = -y;
        x = T(0.5) * (c1.y / y);
    }
    return Complex<T>(x, y);
}

template <typename T>
inline __host__ __device__ Complex<T> pow(const Complex<T> &c1, const T &n) {
    const T r = pow(abs(c1), n);
    const T fi = atan2(c1.y, c1.x);
    return Complex<T>(cos(n * fi) * r, sin(n * fi) * r);
}

template <typename T>
inline __host__ __device__ float Float(const Complex<T> &c1)
{ return (float)c1.x; }

template <typename T>
inline __host__ __device__ double Double(const Complex<T> &c1)
{ return (double)c1.x; }

template <typename T>
#if __CUDACC_VER_MAJOR__ >= 9
inline __device__ Complex<T> __shfl_down_sync(unsigned mask, const Complex<T> &c1, const unsigned delta) {
    Complex<T> ret;
    ret.x = __shfl_down_sync(0xFFFFFF, c1.x, delta);
    ret.y = __shfl_down_sync(0xFFFFFF, c1.y, delta);
    return ret;
}
#else
inline __device__ Complex<T> __shfl_down(const Complex<T> &c1, const unsigned delta) {

    Complex<T> ret;
    ret.x = __shfl_down(c1.x, delta);
    ret.y = __shfl_down(c1.y, delta);
    return ret;
}
#endif

template <typename T>
#if __CUDACC_VER_MAJOR__ >= 9
inline __device__ Complex<T> __shfl_sync(unsigned mask, const Complex<T> &c1, const int src) {
    Complex<T> ret;
    ret.x = __shfl_sync(0xFFFFFF, c1.x, src);
    ret.y = __shfl_sync(0xFFFFFF, c1.y, src);
    return ret;
}
#else
inline __device__ Complex<T> __shfl(const Complex<T> &c1, const int src) {

    Complex<T> ret;
    ret.x = __shfl(c1.x, src);
    ret.y = __shfl(c1.y, src);
    return ret;
}
#endif
// Arithmetic operators

/// operator +
template <typename T>
inline __host__ __device__ Complex<T> operator+(const T &left, const Complex<T> &right)
{ return Complex<T>(left + right.x, right.y); }

/// operator +
template <typename T>
inline __host__ __device__ Complex<T> operator+(const Complex<T> &left, const T &right)
{ return Complex<T>(left.x + right, left.y); }

/// operator +
template <typename T>
inline __host__ __device__ Complex<T> operator+(const Complex<T> &left, const Complex<T> &right)
{ return Complex<T>(left.x + right.x, left.y + right.y); }

/// operator +
template <typename T>
inline __host__ __device__ Complex<T> operator+(const Complex<T> &right)
{ return right; }

/// operator -
template <typename T>
inline __host__ __device__ Complex<T> operator-(const T &left, const Complex<T> &right)
{ return Complex<T>(left - right.x, -right.y); }

/// operator -
template <typename T>
inline __host__ __device__ Complex<T> operator-(const Complex<T> &left, const T &right)
{ return Complex<T>(left.x - right, left.y); }

/// operator -
template <typename T>
inline __host__ __device__ Complex<T> operator-(const Complex<T> &left, const Complex<T> &right)
{ return Complex<T>(left.x - right.x, left.y - right.y); }

/// operator -
template <typename T>
inline __host__ __device__ Complex<T> operator-(const Complex<T> &right)
{ return Complex<T>(-right.x, -right.y); }

/// operator *
template <typename T>
inline __host__ __device__ Complex<T> operator*(const T &left, const Complex<T> &right)
{ return Complex<T>(right.x * left, right.y * left); }

/// operator *
template <typename T>
inline __host__ __device__ Complex<T> operator*(Complex<T> &left, const T &right)
{ return Complex<T>(left.x * right, left.y * right); }

/// operator *
template <typename T>
inline __host__ __device__ Complex<T> operator*(const Complex<T> &left, const Complex<T> &right)
{ return Complex<T>(left.x * right.x - left.y * right.y, left.y * right.x + right.y * left.x); }

/// operator /
template <typename T>
inline __host__ __device__ Complex<T> operator/(const T &left, const Complex<T> &right) {
    const T down = right.x * right.x + right.y * right.y;
    return Complex<T>((right.x * left) / down, -(right.y * left) / down);
}

/// operator /
template <typename T>
inline __host__ __device__ Complex<T> operator/(const Complex<T> &left, const T& right)
{ return Complex<T>(left.x / right, left.y / right); }

/// operator /
template <typename T>
inline __host__ __device__ Complex<T> operator/(const Complex<T> &left, const Complex<T> &right) {
    const T down = right.x * right.x + right.y * right.y;
    return Complex<T>((left.x * right.x + left.y * right.y) / down, (left.y * right.x - right.y * left.x) / down);
}

/// operator ^
template <typename T>
inline __host__ __device__ Complex<T> operator^(const Complex<T> &left, const T &n)
{ return pow(left, n); }

// assignment operators

/// operator +=
template <typename T>
inline __host__ __device__ Complex<T> &operator+=(Complex<T> &left, const Complex<T> &right)
{ left.x += right.x; left.y += right.y; return left; }

/// operator *=
template <typename T>
inline __host__ __device__ Complex<T> &operator*=(Complex<T> &left, const Complex<T> &right) {
    T oldX = left.x, oldY = left.y;
    left.x = right.x * oldX - right.y * oldY;
    left.y = right.y * oldX + oldY * right.x;
    return left;
}

/// operator -=
template <typename T>
inline __host__ __device__ Complex<T> &operator-=(Complex<T> &left, const Complex<T> &right)
{ left.x -= right.x; left.y -= right.y; return left; }

// comparison operators

/// operator ==
template <typename T>
inline __host__ __device__ bool operator==(const Complex<T> &left, const Complex<T> &right)
{ return (left.x == right.x) && (left.y == right.y); }

/// operator ==
template <typename T, typename CmpT>
inline __host__ __device__ bool operator==(const Complex<T> &left, const CmpT &right)
{ return left == Complex<T>(right); }

/// operator ==
template <typename T, typename CmpT>
inline __host__ __device__ bool operator==(const CmpT &left, const Complex<T> &right)
{ return Complex<T>(left) == right; }

/// operator !=
template <typename T>
inline __host__ __device__ bool operator!=(const Complex<T> &left, const Complex<T> &right)
{ return !(left == right); }

template <>
class Complex<float> {
    typedef float T;
public:
    T x; ///< Real part of complex number.
    T y; ///< Imaginary part of complex number.
public:
    inline __host__ __device__ Complex() {}
    inline __host__ __device__ Complex(T x1) //implicit for conversions
    { x = x1; y = 0; }
    inline __host__ __device__ Complex(T x1, T y1)
    { x = x1; y = y1; }
    template <typename T2>
    explicit inline __host__ __device__ Complex(const Complex<T2> &other)
    { x = (T)real(other); y = (T)imag(other); }
    inline __host__ __device__ Complex &operator=(const Complex<T> &right)
    { x = right.x; y = right.y; return *this; }
    template <typename T2>
    inline __host__ __device__ Complex &operator=(const Complex<T2> &right)
    { x = (T)real(right); y = (T)imag(right); return *this; }
#ifdef HAVE_EXPLICIT_CAST
    explicit
#endif
            inline __host__ __device__ operator T() const
    { return x; }
#ifdef HAVE_EXPLICIT_CAST
    explicit inline __host__ __device__ operator double() const
    { return x; }
#endif
};

template <>
class Complex<double> {
    typedef double T;
public:
    T x; ///< Real part of complex number.
    T y; ///< Imaginary part of complex number.
public:
    inline __host__ __device__ Complex() {}
    inline __host__ __device__ Complex(T x1) //implicit for conversions
    { x = x1; y = 0; }
    inline __host__ __device__ Complex(T x1, T y1)
    { x = x1; y = y1; }
    inline __host__ __device__ Complex(const Complex<float> &other) //implicit for propagation
    { x = (T)real(other); y = (T)imag(other); }
    inline __host__ __device__ Complex &operator=(const Complex<T> &right)
    { x = right.x; y = right.y; return *this; }
#ifdef HAVE_EXPLICIT_CAST
    explicit
#endif
            inline __host__ __device__ operator T() const
    { return x; }
#ifdef HAVE_EXPLICIT_CAST
    explicit inline __host__ __device__ operator float() const
    { return x; }
#endif
};

template <typename T>
inline __host__ __device__ Complex<float> cmpxf(const Complex<T> &c1)
{ return Complex<float>((float)c1.x, (float)c1.y); }

template <typename T>
inline __host__ __device__ Complex<double> cmpxd(const Complex<T> &c1)
{ return Complex<double>((double)c1.x, (double)c1.y); }


/// Atomic operations
//Atomic ADD
__device__ __inline__ float __dvmh_atomic_add(float *addr, float val) { return atomicAdd(addr, val); }
__device__ __inline__ int __dvmh_atomic_add(int *addr, int val) { return atomicAdd(addr, val); }
__device__ __inline__ unsigned int __dvmh_atomic_add(unsigned int *addr, unsigned int val) { return atomicAdd(addr, val); }
__device__ __inline__ unsigned long long __dvmh_atomic_add(unsigned long long *addr, unsigned long long val) { return atomicAdd(addr, val); }

__device__ __inline__ double __dvmh_atomic_add(double *addr, double val) 
{
#if (__CUDA_ARCH__ < 600)
    unsigned long long* address_as_ull = (unsigned long long*)addr;
    unsigned long long old = *address_as_ull, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    return atomicAdd(addr, val);
#endif
}

//Atomic MIN, MAX
__device__ __inline__ int __dvmh_atomic_min(int *addr, int val) { return atomicMin(addr, val); }
__device__ __inline__ unsigned int __dvmh_atomic_min(unsigned int *addr, unsigned int val) { return atomicMin(addr, val); }

__device__ __inline__ int __dvmh_atomic_max(int *addr, int val) { return atomicMax(addr, val); }
__device__ __inline__ unsigned int __dvmh_atomic_max(unsigned int *addr, unsigned int val) { return atomicMax(addr, val); }

__device__ __inline__ unsigned long long __dvmh_atomic_min(unsigned long long *addr, unsigned long long val) 
{
#if (__CUDA_ARCH__ < 350)
    unsigned long long old = *addr, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr, assumed, min(val, assumed));
    } while (assumed != old);
    return old;
#else   
    return atomicMin(addr, val); 
#endif
}

__device__ __inline__ long long __dvmh_atomic_min(long long *addr, long long val) 
{
#if (__CUDA_ARCH__ < 350)
    long long old = *addr, assumed;
    do 
    {
        assumed = old;
        old = (long long)atomicCAS((unsigned long long*)addr, (unsigned long long)assumed, (unsigned long long)min(val, assumed));
    } while (assumed != old);
    return old;
#else
    return atomicMin(addr, val); 
#endif
}

__device__ __inline__ unsigned long long __dvmh_atomic_max(unsigned long long *addr, unsigned long long val) 
{
#if (__CUDA_ARCH__ < 350)
    unsigned long long old = *addr, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr, assumed, max(val, assumed));
    } while (assumed != old);
    return old;
#else
    return atomicMax(addr, val); 
#endif
}

__device__ __inline__ long long __dvmh_atomic_max(long long *addr, long long val) 
{
#if (__CUDA_ARCH__ < 350)
    long long old = *addr, assumed;
    do 
    {
        assumed = old;
        old = (long long)atomicCAS((unsigned long long*)addr, (unsigned long long)assumed, (unsigned long long)max(val, assumed));
    } while (assumed != old);
    return old;
#else
    return atomicMax(addr, val); 
#endif
}

__device__ __inline__ float __dvmh_atomic_min(float *addr, float val) 
{
    float old = (val >= 0) ? 
                __int_as_float(atomicMin((int *)addr, __float_as_int(val))) : 
                __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(val)));
    return old;
}

__device__ __inline__ float __dvmh_atomic_max(float *addr, float val) 
{
    float old = (val >= 0) ? 
                __int_as_float(atomicMax((int *)addr, __float_as_int(val))) : 
                __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(val)));
    return old;
}

__device__ __inline__ double __dvmh_atomic_min(double *addr, double val) 
{
    double old = (val >= 0) ? 
                 __longlong_as_double(__dvmh_atomic_min((long long *)addr, __double_as_longlong(val))) : 
                 __longlong_as_double(__dvmh_atomic_max((unsigned long long *)addr, (unsigned long long) __double_as_longlong(val)));
    return old;
}

__device__ __inline__ double __dvmh_atomic_max(double *addr, double val) 
{
    double old = (val >= 0) ? 
                 __longlong_as_double(__dvmh_atomic_max((long long *)addr, __double_as_longlong(val))) : 
                 __longlong_as_double(__dvmh_atomic_min((unsigned long long *)addr, (unsigned long long) __double_as_longlong(val)));
    return old;
}

//Atomic prod, TODO
__device__ __inline__ int __dvmh_atomic_prod(int *addr, int val) 
{
    int old = *addr, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr, assumed, val * assumed);
    } while (assumed != old);
    return old;
}

__device__ __inline__ long long __dvmh_atomic_prod(unsigned long long *addr, unsigned long long val) 
{
    unsigned long long old = *addr, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr, assumed, val * assumed);
    } while (assumed != old);
    return old;
}

__device__ __inline__ float __dvmh_atomic_prod(float *addr, float val) { return 0; }
__device__ __inline__ float __dvmh_atomic_prod(double *addr, double val) { return 0; }

//Atomic Bitwise
__device__ __inline__ int __dvmh_atomic_and(int *addr, int val) { return atomicAnd(addr, val); }
__device__ __inline__ int __dvmh_atomic_or(int *addr, int val) { return atomicOr(addr, val); }
__device__ __inline__ int __dvmh_atomic_neqv(int *addr, int val) 
{
    int old = *addr, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr, assumed, (old != val));
    } while (assumed != old);
    return old;
}

__device__ __inline__ int __dvmh_atomic_eqv(int *addr, int val) 
{
    int old = *addr, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr, assumed, (old == val));
    } while (assumed != old);
    return old;
}

//simple XOR_SHIFT random (rng_xor128)
template<typename dType>
__device__ __inline__ void __dvmh_rand(dType &retVal, uint4 &state)
{
    unsigned int t;
    t = state.x ^ (state.x << 11);

    state.x = state.y;
    state.y = state.z;
    state.z = state.w;

    state.w = (state.w ^ (state.w >> 19)) ^ (t ^ (t >> 8));
    retVal = (dType)state.w / (dType)UINT_MAX;
}
