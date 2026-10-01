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
