// kiln's CUDA prelude. Under NVRTC (__CUDACC_RTC__) this is plain CUDA.
// Everywhere else the same kernel source compiles as C++ against an
// emulator of the CUDA execution model: a block's threads are OS threads,
// __syncthreads is a barrier over them, and warp operations (shuffles,
// tensor-core mma) exchange values through a per-warp buffer between two
// warp barriers. Blocks run one after another, so shared memory is one
// buffer reused by every block. Kernels are written so that every thread
// reaches every barrier and every warp operation, as CUDA requires.
#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define KILN_CUDA 1
#define KDEV __device__ __forceinline__
#define KGLOBAL(n) extern "C" __global__ void __launch_bounds__(n)
#define SMEM extern __shared__ __align__(16) unsigned char kiln_smem[]
#define KSYNC() __syncthreads()
#define KWSYNC() __syncwarp()
#define KINF __int_as_float(0x7f800000)
#define KNAN __int_as_float(0x7fffffff)
KDEV float kshfl_xor(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
// fp16 as raw bits (NVRTC has no cuda_fp16.h without the toolkit's headers).
typedef unsigned short khalf;
KDEV khalf kf2h(float f) {
  khalf h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
KDEV float kh2f(khalf h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
// Two consecutive halves as one 32-bit register (p 4-byte aligned).
KDEV unsigned kld2(const khalf *p) { return *(const unsigned *)p; }
// D = A B + D on tensor cores: A 16x8 (row), B 8x8 (col), f16 in, f32
// accumulate. Fragments, with g = lane / 4 and t = lane % 4:
//   a0 = A[g][2t..2t+1], a1 = A[g+8][2t..2t+1], b0 = B[2t..2t+1][g],
//   c[0..1] = C[g][2t..2t+1], c[2..3] = C[g+8][2t..2t+1].
KDEV void kmma(float *c, unsigned a0, unsigned a1, unsigned b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}
#else
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define KDEV static inline
#define KGLOBAL(n) static void
#define __restrict__ __restrict
#define KINF INFINITY
#define KNAN NAN
struct kdim3 {
  unsigned x, y, z;
};
#define KEMU_XCH 64
struct KTeam {
  pthread_barrier_t block;
  pthread_barrier_t warp[32];
  float xch[32][32 * KEMU_XCH];
  void (*fn)(float **);
  float **args;
  kdim3 grid, dim;
  unsigned char *smem;
};
static thread_local kdim3 threadIdx, blockIdx, blockDim, gridDim;
static thread_local KTeam *kteam;
static thread_local unsigned ktid;
#define SMEM unsigned char *kiln_smem = kteam->smem
static inline void KSYNC() { pthread_barrier_wait(&kteam->block); }
static inline void KWSYNC() { pthread_barrier_wait(&kteam->warp[ktid / 32]); }
static inline float *kxch() { return kteam->xch[ktid / 32]; }
static inline float kshfl_xor(float v, int m) {
  float *x = kxch();
  unsigned l = ktid % 32;
  x[l] = v;
  KWSYNC();
  float r = x[l ^ m];
  KWSYNC();
  return r;
}
typedef unsigned short khalf;
// IEEE binary16, round to nearest even (as cvt.rn.f16.f32).
static inline khalf kf2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
  int32_t e = (int32_t)((x >> 23) & 0xff);
  if (e == 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  int32_t exp = e - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    uint32_t shift = 14 - exp, h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return sign | h;
  }
  uint32_t h = ((uint32_t)exp << 10) | (mant >> 13), rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
  return sign | h;
}
static inline float kh2f(khalf h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      int32_t e = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        e--;
      }
      x = sign | ((uint32_t)(e + 127 - 15) << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7f800000 | (mant << 13);
  } else {
    x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}
static inline unsigned kld2(const khalf *p) {
  unsigned v;
  memcpy(&v, p, 4);
  return v;
}
// The warp's fragments meet in the exchange buffer; each lane computes its
// four outputs from them, in the fragment layout of mma.m16n8k8.
static inline void kmma(float *c, unsigned a0, unsigned a1, unsigned b0) {
  unsigned *x = (unsigned *)kxch();
  unsigned l = ktid % 32, g = l / 4, t = l % 4;
  x[l * 3] = a0;
  x[l * 3 + 1] = a1;
  x[l * 3 + 2] = b0;
  KWSYNC();
  for (int i = 0; i < 4; i++) {
    unsigned row = g + (i >= 2 ? 8 : 0), col = 2 * t + (i & 1);
    float s = 0.0f;
    for (unsigned k = 0; k < 8; k++) {
      unsigned ra = x[((row % 8) * 4 + k / 2) * 3 + (row >= 8 ? 1 : 0)];
      unsigned rb = x[(col * 4 + k / 2) * 3 + 2];
      khalf ha = (khalf)(ra >> (16 * (k & 1))), hb = (khalf)(rb >> (16 * (k & 1)));
      s += kh2f(ha) * kh2f(hb);
    }
    c[i] += s;
  }
  KWSYNC();
}
struct KStart {
  KTeam *t;
  unsigned tid;
};
static void *kemu_run(void *p) {
  KStart *s = (KStart *)p;
  kteam = s->t;
  ktid = s->tid;
  blockDim = kteam->dim;
  gridDim = kteam->grid;
  threadIdx.x = ktid % blockDim.x;
  threadIdx.y = ktid / blockDim.x;
  threadIdx.z = 0;
  for (unsigned z = 0; z < gridDim.z; z++)
    for (unsigned y = 0; y < gridDim.y; y++)
      for (unsigned x = 0; x < gridDim.x; x++) {
        blockIdx.x = x;
        blockIdx.y = y;
        blockIdx.z = z;
        kteam->fn(kteam->args);
        // Every thread leaves the block before the next one reuses shared memory.
        pthread_barrier_wait(&kteam->block);
      }
  return 0;
}
static void kemu_launch(void (*fn)(float **), float **args, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned smem) {
  unsigned n = bx * by;
  KTeam *t = (KTeam *)calloc(1, sizeof(KTeam));
  t->fn = fn;
  t->args = args;
  t->grid = {gx, gy, gz};
  t->dim = {bx, by, 1};
  t->smem = (unsigned char *)calloc(smem + 16, 1);
  // Uninitialized shared memory is NaN, so a read before a write shows up.
  for (unsigned i = 0; i < smem / 4; i++) ((float *)t->smem)[i] = NAN;
  pthread_barrier_init(&t->block, 0, n);
  for (unsigned w = 0; w < (n + 31) / 32; w++) pthread_barrier_init(&t->warp[w], 0, 32);
  pthread_t *th = (pthread_t *)malloc(n * sizeof(pthread_t));
  KStart *st = (KStart *)malloc(n * sizeof(KStart));
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1 << 18);
  for (unsigned i = 0; i < n; i++) {
    st[i] = {t, i};
    pthread_create(&th[i], &attr, kemu_run, &st[i]);
  }
  for (unsigned i = 0; i < n; i++) pthread_join(th[i], 0);
  pthread_attr_destroy(&attr);
  pthread_barrier_destroy(&t->block);
  for (unsigned w = 0; w < (n + 31) / 32; w++) pthread_barrier_destroy(&t->warp[w]);
  free(th);
  free(st);
  free(t->smem);
  free(t);
}
#endif

// ---- shared by both: math and reductions ----
KDEV float kmax(float a, float b) { return a > b ? a : b; }
KDEV float ksig(float x) { return 1.0f / (1.0f + expf(-x)); }
KDEV float krelu(float x) { return x > 0.0f ? x : 0.0f; }
// Reduce over groups of g consecutive lanes (g a power of two, <= 32).
KDEV float kwarp_sum(float v, int g) {
  for (int m = g / 2; m > 0; m /= 2) v += kshfl_xor(v, m);
  return v;
}
KDEV float kwarp_max(float v, int g) {
  for (int m = g / 2; m > 0; m /= 2) v = kmax(v, kshfl_xor(v, m));
  return v;
}
