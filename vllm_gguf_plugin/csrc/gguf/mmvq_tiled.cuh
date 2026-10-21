// Tensor-core GEMV for small batches on a tiled weight layout ("tiled" kernels).
//
// The GGUF row-major layout is repacked once at load time into tiles of 16 rows:
//   tile t, superblock i  ->  16 rows x ROWB bytes, contiguous and 16-byte aligned
// so a warp can stream a whole tile with 16-byte cp.async copies through a shared-memory pipeline,
// and one int8 mma instruction (16 rows x 8 tokens x 16/32 weights) covers up to 8 input vectors.

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// mma.sync.aligned.m16n8k16 (s8): A 16x16 row-major (a0: row g, k 4t..4t+3; a1: row g+8), B 16x8 col-major
// (b0: k 4t..4t+3, n g), C 16x8 (c0,c1: row g, n 2t,2t+1; c2,c3: row g+8). g = lane >> 2, t = lane & 3.
__device__ __forceinline__ void mma_s8_16x8x16(int * c, const int a0, const int a1, const int b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#endif
}

#define Q3T_ROWB 112      // hmask[32] | qs[64] | scales as int8[16]  (scales are decoded when repacking)
#define Q3T_TILE_BYTES (16 * Q3T_ROWB)
#define Q3T_D_BYTES 32    // per tile and superblock: 16 x half d
#define Q3T_STAGE_BYTES (Q3T_TILE_BYTES + Q3T_D_BYTES)
#ifndef Q3T_WARPS
#define Q3T_WARPS 4
#endif
#ifndef Q3T_STAGES
#define Q3T_STAGES 6
#endif

template <typename scalar_t>
static __global__ void mul_mat_q3k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int wk = threadIdx.y;
    const int row0 = blockIdx.x * 16;
    const int vec0 = blockIdx.y * 8;
    const int nv = min(8, nvecs - vec0);
    const int bpr = ncols / 256;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int nsb = (bpr - wk + Q3T_WARPS - 1) / Q3T_WARPS;     // superblocks of this warp: wk, wk+WARPS, ...

    __shared__ __align__(16) uint8_t smem[Q3T_WARPS][Q3T_STAGES][Q3T_STAGE_BYTES];
    const uint8_t * tile_base = wt + (size_t) blockIdx.x * bpr * Q3T_TILE_BYTES;
    const uint8_t * d_base = wd + (size_t) blockIdx.x * bpr * Q3T_D_BYTES;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int tokB = vec0 + min(g, nv - 1);
    const int n0 = min(2*t, nv - 1), n1 = min(2*t + 1, nv - 1);
    const block_q8_1 * yB  = y + (size_t) tokB*(nrows_y/QK8_1);
    const block_q8_1 * y0  = y + (size_t) (vec0 + n0)*(nrows_y/QK8_1);
    const block_q8_1 * y1  = y + (size_t) (vec0 + n1)*(nrows_y/QK8_1);

    auto issue = [&](int n) {
        const int i = wk + n*Q3T_WARPS;
        const uint8_t * src = tile_base + (size_t) i * Q3T_TILE_BYTES;
        uint8_t * dstp = smem[wk][n % Q3T_STAGES];
#pragma unroll
        for (int c = lane; c < Q3T_TILE_BYTES/16; c += 32) {
            cp_async16(dstp + c*16, src + c*16);
        }
        if (lane < Q3T_D_BYTES/16) {
            cp_async16(dstp + Q3T_TILE_BYTES + lane*16, d_base + (size_t) i * Q3T_D_BYTES + lane*16);
        }
    };

#pragma unroll
    for (int s = 0; s < Q3T_STAGES - 1; ++s) {
        if (s < nsb) {
            issue(s);
        }
        cp_async_commit();
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < nsb; ++n) {
        cp_async_wait<Q3T_STAGES - 2>();
        __syncwarp();
        if (n + Q3T_STAGES - 1 < nsb) {
            issue(n + Q3T_STAGES - 1);
        }
        cp_async_commit();

        const int i = wk + n*Q3T_WARPS;
        const uint8_t * stage = smem[wk][n % Q3T_STAGES];
        const uint8_t * bufA = stage + g * Q3T_ROWB;
        const uint8_t * bufB = bufA + 8 * Q3T_ROWB;
        const uint32_t * hmA = (const uint32_t *) bufA;          // hmask[32]
        const uint32_t * hmB = (const uint32_t *) bufB;
        const uint32_t * qsA = (const uint32_t *) (bufA + 32);   // qs[64]
        const uint32_t * qsB = (const uint32_t *) (bufB + 32);
        const uint32_t * scA = (const uint32_t *) (bufA + 96);   // scales int8[16]
        const uint32_t * scB = (const uint32_t *) (bufB + 96);
        const half * dh = (const half *) (stage + Q3T_TILE_BYTES);
        const float dA = __half2float(dh[g]);
        const float dB = __half2float(dh[g + 8]);

        const uint32_t hA[2] = {hmA[t], hmA[4 + t]};
        const uint32_t hB[2] = {hmB[t], hmB[4 + t]};
        const uint32_t sA4[4] = {scA[0], scA[1], scA[2], scA[3]};
        const uint32_t sB4[4] = {scB[0], scB[1], scB[2], scB[3]};

        float sb[4] = {0.f, 0.f, 0.f, 0.f};   // this superblock, before the row scale d

#pragma unroll
        for (int h = 0; h < 2; ++h) {            // half of the superblock (weights 0..127 / 128..255)
            uint32_t qA[2], qB[2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {        // first / second 16 weights of a 32-group
                qA[p] = qsA[h*8 + p*4 + t];
                qB[p] = qsB[h*8 + p*4 + t];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {        // 32-group within the half: 2-bit shift j
                const int grp = h*4 + j;         // 32-group index 0..7 (also hmask bit)
                const float dy0 = __low2float(y0[i*8 + grp].ds);
                const float dy1 = __low2float(y1[i*8 + grp].ds);
                const int * q8 = (const int *) yB[i*8 + grp].qs;
                int cc[2][4];
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int vilA = (qA[p] >> (2*j)) & 0x03030303;
                    const int vihA = (((~hA[p]) >> grp) << 2) & 0x04040404;
                    const int vilB = (qB[p] >> (2*j)) & 0x03030303;
                    const int vihB = (((~hB[p]) >> grp) << 2) & 0x04040404;
                    cc[p][0] = cc[p][1] = cc[p][2] = cc[p][3] = 0;
                    mma_s8_16x8x16(cc[p], __vsubss4(vilA, vihA), __vsubss4(vilB, vihB), q8[p*4 + t]);
                }
                const int s0 = grp*2, s1 = grp*2 + 1;
                const int a0 = ((int) (sA4[s0 >> 2] << (24 - 8*(s0 & 3)))) >> 24;
                const int a1 = ((int) (sA4[s1 >> 2] << (24 - 8*(s1 & 3)))) >> 24;
                const int b0 = ((int) (sB4[s0 >> 2] << (24 - 8*(s0 & 3)))) >> 24;
                const int b1 = ((int) (sB4[s1 >> 2] << (24 - 8*(s1 & 3)))) >> 24;
                sb[0] += (float) (cc[0][0]*a0 + cc[1][0]*a1) * dy0;
                sb[1] += (float) (cc[0][1]*a0 + cc[1][1]*a1) * dy1;
                sb[2] += (float) (cc[0][2]*b0 + cc[1][2]*b1) * dy0;
                sb[3] += (float) (cc[0][3]*b0 + cc[1][3]*b1) * dy1;
            }
        }
        acc[0] += sb[0] * dA;
        acc[1] += sb[1] * dA;
        acc[2] += sb[2] * dB;
        acc[3] += sb[3] * dB;
    }
    cp_async_wait<0>();
    __syncthreads();

    // sum the K-split partial results of the warps (reuse stage memory)
    float (*red)[32][4] = (float (*)[32][4]) &smem[0][0][0];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[wk][lane][k] = acc[k];
    }
    __syncthreads();
    if (wk != 0) {
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int w = 1; w < Q3T_WARPS; ++w) {
            acc[k] += red[w][lane][k];
        }
    }
    const int r0 = row0 + g, r1 = row0 + g + 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int tok = 2*t + u;
        if (tok < nv) {
            if (r0 < nrows) dst[(size_t)(vec0 + tok)*nrows + r0] = acc[u];
            if (r1 < nrows) dst[(size_t)(vec0 + tok)*nrows + r1] = acc[2 + u];
        }
    }
}

// Dequantize the tiled Q3_K layout to a dense [rows, cols] matrix (for batches too large for the mma kernel).
template <typename scalar_t>
static __global__ void dequantize_q3k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const int tid = threadIdx.x;               // 0..255: row = tid / 16, sub-block s = tid % 16
    const int r = tid >> 4;
    const int s = tid & 15;
    const int row = tile * 16 + r;
    if (row >= rows) {
        return;
    }
    const uint8_t * blk = wt + ((size_t) tile * bpr + i) * Q3T_TILE_BYTES + r * Q3T_ROWB;
    const uint8_t * hm = blk;
    const uint8_t * qs = blk + 32;
    const int8_t sc = ((const int8_t *) (blk + 96))[s];
    const float d = __half2float(((const half *) (wd + ((size_t) tile * bpr + i) * Q3T_D_BYTES))[r]);
    const int grp = s >> 1;
    const int half_ = grp >> 2;
    const int j = grp & 3;
    scalar_t * o = out + (size_t) row * cols + i * 256 + 16 * s;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int pos = (s & 1) * 16 + k;
        const int q = (qs[half_ * 32 + pos] >> (2 * j)) & 3;
        const int h = (hm[pos] >> grp) & 1;
        o[k] = (scalar_t) (d * sc * (q - (h ? 0 : 4)));
    }
}


// mma.sync.aligned.m16n8k32 (s8): A 16x32 (a0: row g, k 4t..; a1: row g+8; a2: row g, k 16+4t..; a3: row g+8), B 32x8 (b0: k 4t.., b1: k 16+4t..)
__device__ __forceinline__ void mma_s8_16x8x32_t(int * c, const int a0, const int a1, const int a2, const int a3, const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// ---------------------------------------------------------------------------------------------
// Q4_K: tiled row-block = qs[128] | sc[8] (uint8) | mn[8] (uint8)  = 144 bytes; d plane: 16 x (half d, half dmin)
#define Q4T_ROWB 144
#define Q4T_TILE_BYTES (16 * Q4T_ROWB)
#define Q4T_D_BYTES 64
#define Q4T_STAGE_BYTES (Q4T_TILE_BYTES + Q4T_D_BYTES)
#define Q4T_WARPS 4
#define Q4T_STAGES 5

template <typename scalar_t>
static __global__ void mul_mat_q4k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int wk = threadIdx.y;
    const int row0 = blockIdx.x * 16;
    const int vec0 = blockIdx.y * 8;
    const int nv = min(8, nvecs - vec0);
    const int bpr = ncols / 256;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int nsb = (bpr - wk + Q4T_WARPS - 1) / Q4T_WARPS;

    __shared__ __align__(16) uint8_t smem[Q4T_WARPS][Q4T_STAGES][Q4T_STAGE_BYTES];
    const uint8_t * tile_base = wt + (size_t) blockIdx.x * bpr * Q4T_TILE_BYTES;
    const uint8_t * d_base = wd + (size_t) blockIdx.x * bpr * Q4T_D_BYTES;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int tokB = vec0 + min(g, nv - 1);
    const int n0 = min(2*t, nv - 1), n1 = min(2*t + 1, nv - 1);
    const block_q8_1 * yB = y + (size_t) tokB*(nrows_y/QK8_1);
    const block_q8_1 * y0 = y + (size_t) (vec0 + n0)*(nrows_y/QK8_1);
    const block_q8_1 * y1 = y + (size_t) (vec0 + n1)*(nrows_y/QK8_1);

    auto issue = [&](int n) {
        const int i = wk + n*Q4T_WARPS;
        const uint8_t * src = tile_base + (size_t) i * Q4T_TILE_BYTES;
        uint8_t * dstp = smem[wk][n % Q4T_STAGES];
#pragma unroll
        for (int c = lane; c < Q4T_TILE_BYTES/16; c += 32) {
            cp_async16(dstp + c*16, src + c*16);
        }
        if (lane < Q4T_D_BYTES/16) {
            cp_async16(dstp + Q4T_TILE_BYTES + lane*16, d_base + (size_t) i * Q4T_D_BYTES + lane*16);
        }
    };

#pragma unroll
    for (int s = 0; s < Q4T_STAGES - 1; ++s) {
        if (s < nsb) {
            issue(s);
        }
        cp_async_commit();
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < nsb; ++n) {
        cp_async_wait<Q4T_STAGES - 2>();
        __syncwarp();
        if (n + Q4T_STAGES - 1 < nsb) {
            issue(n + Q4T_STAGES - 1);
        }
        cp_async_commit();

        const int i = wk + n*Q4T_WARPS;
        const uint8_t * stage = smem[wk][n % Q4T_STAGES];
        const uint8_t * bufA = stage + g * Q4T_ROWB;
        const uint8_t * bufB = bufA + 8 * Q4T_ROWB;
        const uint32_t * qsA = (const uint32_t *) bufA;
        const uint32_t * qsB = (const uint32_t *) bufB;
        const uint32_t * scA = (const uint32_t *) (bufA + 128);   // sc[8] | mn[8]
        const uint32_t * scB = (const uint32_t *) (bufB + 128);
        const half2 * dm = (const half2 *) (stage + Q4T_TILE_BYTES);
        const float2 dmA = __half22float2(dm[g]);
        const float2 dmB = __half22float2(dm[g + 8]);
        const uint32_t sA4[4] = {scA[0], scA[1], scA[2], scA[3]};
        const uint32_t sB4[4] = {scB[0], scB[1], scB[2], scB[3]};

        float sbm[4] = {0.f, 0.f, 0.f, 0.f};   // sum_j sc_j * (q . y) * dy
        float sbn[4] = {0.f, 0.f, 0.f, 0.f};   // sum_j mn_j * sum(y)

#pragma unroll
        for (int p = 0; p < 4; ++p) {              // pair of sub-blocks: low nibble 2p, high nibble 2p+1
            const uint32_t wA0 = qsA[8*p + t], wA1 = qsA[8*p + 4 + t];
            const uint32_t wB0 = qsB[8*p + t], wB1 = qsB[8*p + 4 + t];
#pragma unroll
            for (int hl = 0; hl < 2; ++hl) {
                const int j = 2*p + hl;
                const uint32_t mask = 0x0F0F0F0F;
                const int a0 = hl ? (int) ((wA0 >> 4) & mask) : (int) (wA0 & mask);
                const int a1 = hl ? (int) ((wB0 >> 4) & mask) : (int) (wB0 & mask);
                const int a2 = hl ? (int) ((wA1 >> 4) & mask) : (int) (wA1 & mask);
                const int a3 = hl ? (int) ((wB1 >> 4) & mask) : (int) (wB1 & mask);
                const block_q8_1 * yb = &yB[i*8 + j];
                const int * q8 = (const int *) yb->qs;
                int c[4] = {0, 0, 0, 0};
                mma_s8_16x8x32_t(c, a0, a1, a2, a3, q8[t], q8[4 + t]);
                // sum of the int8 activations (against an all-ones A): consistent with the rounded values used above
                int cs[4] = {0, 0, 0, 0};
                mma_s8_16x8x32_t(cs, 0x01010101, 0x01010101, 0x01010101, 0x01010101, q8[t], q8[4 + t]);
                const float dy0 = __low2float(y0[i*8 + j].ds);
                const float dy1 = __low2float(y1[i*8 + j].ds);
                const float sy0 = dy0 * (float) cs[0];
                const float sy1 = dy1 * (float) cs[1];
                const int scAj = (sA4[j >> 2] >> (8*(j & 3))) & 0xFF;
                const int mnAj = (sA4[2 + (j >> 2)] >> (8*(j & 3))) & 0xFF;
                const int scBj = (sB4[j >> 2] >> (8*(j & 3))) & 0xFF;
                const int mnBj = (sB4[2 + (j >> 2)] >> (8*(j & 3))) & 0xFF;
                sbm[0] += (float) (c[0]*scAj) * dy0;
                sbm[1] += (float) (c[1]*scAj) * dy1;
                sbm[2] += (float) (c[2]*scBj) * dy0;
                sbm[3] += (float) (c[3]*scBj) * dy1;
                sbn[0] += (float) mnAj * sy0;
                sbn[1] += (float) mnAj * sy1;
                sbn[2] += (float) mnBj * sy0;
                sbn[3] += (float) mnBj * sy1;
            }
        }
        acc[0] += dmA.x*sbm[0] - dmA.y*sbn[0];
        acc[1] += dmA.x*sbm[1] - dmA.y*sbn[1];
        acc[2] += dmB.x*sbm[2] - dmB.y*sbn[2];
        acc[3] += dmB.x*sbm[3] - dmB.y*sbn[3];
    }
    cp_async_wait<0>();
    __syncthreads();

    float (*red)[32][4] = (float (*)[32][4]) &smem[0][0][0];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[wk][lane][k] = acc[k];
    }
    __syncthreads();
    if (wk != 0) {
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int w = 1; w < Q4T_WARPS; ++w) {
            acc[k] += red[w][lane][k];
        }
    }
    const int r0 = row0 + g, r1 = row0 + g + 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int tok = 2*t + u;
        if (tok < nv) {
            if (r0 < nrows) dst[(size_t)(vec0 + tok)*nrows + r0] = acc[u];
            if (r1 < nrows) dst[(size_t)(vec0 + tok)*nrows + r1] = acc[2 + u];
        }
    }
}

template <typename scalar_t>
static __global__ void dequantize_q4k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const int tid = threadIdx.x;               // 0..127: row = tid / 8, sub-block j = tid % 8
    const int r = tid >> 3;
    const int j = tid & 7;
    const int row = tile * 16 + r;
    if (row >= rows) {
        return;
    }
    const uint8_t * blk = wt + ((size_t) tile * bpr + i) * Q4T_TILE_BYTES + r * Q4T_ROWB;
    const half2 dm = ((const half2 *) (wd + ((size_t) tile * bpr + i) * Q4T_D_BYTES))[r];
    const float d = __low2float(dm) * blk[128 + j];
    const float m = __high2float(dm) * blk[136 + j];
    const int pr = j >> 1;
    scalar_t * o = out + (size_t) row * cols + i * 256 + 32 * j;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int b = blk[32*pr + k];
        const int q = (j & 1) ? (b >> 4) : (b & 15);
        o[k] = (scalar_t) (d * q - m);
    }
}

// ---------------------------------------------------------------------------------------------
// Q6_K: tiled row-block = ql[128] | qh[64] | sc int8[16] = 208 bytes; d plane: 16 x half
#define Q6T_ROWB 208
#define Q6T_TILE_BYTES (16 * Q6T_ROWB)
#define Q6T_D_BYTES 32
#define Q6T_STAGE_BYTES (Q6T_TILE_BYTES + Q6T_D_BYTES)
#define Q6T_WARPS 4
#define Q6T_STAGES 3

template <typename scalar_t>
static __global__ void mul_mat_q6k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int wk = threadIdx.y;
    const int row0 = blockIdx.x * 16;
    const int vec0 = blockIdx.y * 8;
    const int nv = min(8, nvecs - vec0);
    const int bpr = ncols / 256;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int nsb = (bpr - wk + Q6T_WARPS - 1) / Q6T_WARPS;

    __shared__ __align__(16) uint8_t smem[Q6T_WARPS][Q6T_STAGES][Q6T_STAGE_BYTES];
    const uint8_t * tile_base = wt + (size_t) blockIdx.x * bpr * Q6T_TILE_BYTES;
    const uint8_t * d_base = wd + (size_t) blockIdx.x * bpr * Q6T_D_BYTES;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int tokB = vec0 + min(g, nv - 1);
    const int n0 = min(2*t, nv - 1), n1 = min(2*t + 1, nv - 1);
    const block_q8_1 * yB = y + (size_t) tokB*(nrows_y/QK8_1);
    const block_q8_1 * y0 = y + (size_t) (vec0 + n0)*(nrows_y/QK8_1);
    const block_q8_1 * y1 = y + (size_t) (vec0 + n1)*(nrows_y/QK8_1);

    auto issue = [&](int n) {
        const int i = wk + n*Q6T_WARPS;
        const uint8_t * src = tile_base + (size_t) i * Q6T_TILE_BYTES;
        uint8_t * dstp = smem[wk][n % Q6T_STAGES];
#pragma unroll
        for (int c = lane; c < Q6T_TILE_BYTES/16; c += 32) {
            cp_async16(dstp + c*16, src + c*16);
        }
        if (lane < Q6T_D_BYTES/16) {
            cp_async16(dstp + Q6T_TILE_BYTES + lane*16, d_base + (size_t) i * Q6T_D_BYTES + lane*16);
        }
    };

#pragma unroll
    for (int s = 0; s < Q6T_STAGES - 1; ++s) {
        if (s < nsb) {
            issue(s);
        }
        cp_async_commit();
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < nsb; ++n) {
        cp_async_wait<Q6T_STAGES - 2>();
        __syncwarp();
        if (n + Q6T_STAGES - 1 < nsb) {
            issue(n + Q6T_STAGES - 1);
        }
        cp_async_commit();

        const int i = wk + n*Q6T_WARPS;
        const uint8_t * stage = smem[wk][n % Q6T_STAGES];
        const uint8_t * bufA = stage + g * Q6T_ROWB;
        const uint8_t * bufB = bufA + 8 * Q6T_ROWB;
        const uint32_t * qlA = (const uint32_t *) bufA;           // ql[128]
        const uint32_t * qlB = (const uint32_t *) bufB;
        const uint32_t * qhA = (const uint32_t *) (bufA + 128);   // qh[64]
        const uint32_t * qhB = (const uint32_t *) (bufB + 128);
        const uint32_t * scA = (const uint32_t *) (bufA + 192);   // scales int8[16]
        const uint32_t * scB = (const uint32_t *) (bufB + 192);
        const half * dh = (const half *) (stage + Q6T_TILE_BYTES);
        const float dA = __half2float(dh[g]);
        const float dB = __half2float(dh[g + 8]);
        const uint32_t sA4[4] = {scA[0], scA[1], scA[2], scA[3]};
        const uint32_t sB4[4] = {scB[0], scB[1], scB[2], scB[3]};

        float sb[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int h = 0; h < 2; ++h) {                 // 128-weight half
#pragma unroll
            for (int lh = 0; lh < 2; ++lh) {          // 16 of the 32 lanes l
                // words for k = 4t..4t+3 of this 16-weight run: l = 16*lh + 4t
                const int wl = (h*64 + 16*lh)/4 + t;       // ql word for u&1 == 0 (bytes h*64 + l)
                const int wl2 = wl + 8;                    // ql word for u&1 == 1 (bytes h*64 + 32 + l)
                const int wh = (h*32 + 16*lh)/4 + t;       // qh word
                const uint32_t qa0 = qlA[wl], qa1 = qlA[wl2], ha = qhA[wh];
                const uint32_t qb0 = qlB[wl], qb1 = qlB[wl2], hb = qhB[wh];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint32_t qa = (u & 1) ? qa1 : qa0;
                    const uint32_t qb = (u & 1) ? qb1 : qb0;
                    const uint32_t nibA = (u < 2) ? (qa & 0x0F0F0F0F) : ((qa >> 4) & 0x0F0F0F0F);
                    const uint32_t nibB = (u < 2) ? (qb & 0x0F0F0F0F) : ((qb >> 4) & 0x0F0F0F0F);
                    const uint32_t hiA = ((ha >> (2*u)) & 0x03030303) << 4;
                    const uint32_t hiB = ((hb >> (2*u)) & 0x03030303) << 4;
                    const int vA = __vsubss4((int) (nibA | hiA), 0x20202020);
                    const int vB = __vsubss4((int) (nibB | hiB), 0x20202020);
                    const int yb = 4*h + u;                // 32-weight activation block
                    const int * q8 = (const int *) yB[i*8 + yb].qs;
                    int cc[4] = {0, 0, 0, 0};
                    mma_s8_16x8x16(cc, vA, vB, q8[lh*4 + t]);
                    const int s = 8*h + lh + 2*u;
                    const int scAs = ((int) (sA4[s >> 2] << (24 - 8*(s & 3)))) >> 24;
                    const int scBs = ((int) (sB4[s >> 2] << (24 - 8*(s & 3)))) >> 24;
                    const float dy0 = __low2float(y0[i*8 + yb].ds);
                    const float dy1 = __low2float(y1[i*8 + yb].ds);
                    sb[0] += (float) (cc[0]*scAs) * dy0;
                    sb[1] += (float) (cc[1]*scAs) * dy1;
                    sb[2] += (float) (cc[2]*scBs) * dy0;
                    sb[3] += (float) (cc[3]*scBs) * dy1;
                }
            }
        }
        acc[0] += sb[0] * dA;
        acc[1] += sb[1] * dA;
        acc[2] += sb[2] * dB;
        acc[3] += sb[3] * dB;
    }
    cp_async_wait<0>();
    __syncthreads();

    float (*red)[32][4] = (float (*)[32][4]) &smem[0][0][0];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[wk][lane][k] = acc[k];
    }
    __syncthreads();
    if (wk != 0) {
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int w = 1; w < Q6T_WARPS; ++w) {
            acc[k] += red[w][lane][k];
        }
    }
    const int r0 = row0 + g, r1 = row0 + g + 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int tok = 2*t + u;
        if (tok < nv) {
            if (r0 < nrows) dst[(size_t)(vec0 + tok)*nrows + r0] = acc[u];
            if (r1 < nrows) dst[(size_t)(vec0 + tok)*nrows + r1] = acc[2 + u];
        }
    }
}

template <typename scalar_t>
static __global__ void dequantize_q6k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const int tid = threadIdx.x;               // 0..255: row = tid / 16, sub-block s = tid % 16
    const int r = tid >> 4;
    const int s = tid & 15;
    const int row = tile * 16 + r;
    if (row >= rows) {
        return;
    }
    const uint8_t * blk = wt + ((size_t) tile * bpr + i) * Q6T_TILE_BYTES + r * Q6T_ROWB;
    const float d = __half2float(((const half *) (wd + ((size_t) tile * bpr + i) * Q6T_D_BYTES))[r]);
    const float sc = d * ((const int8_t *) (blk + 192))[s];
    const int h = s >> 3, lh = s & 1, u = (s >> 1) & 3;      // s = 8h + lh + 2u
    scalar_t * o = out + (size_t) row * cols + i * 256 + 128*h + 32*u + 16*lh;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int l = 16*lh + k;
        const int qlb = blk[h*64 + (u & 1)*32 + l];
        const int nib = (u < 2) ? (qlb & 15) : (qlb >> 4);
        const int hi = (blk[128 + h*32 + l] >> (2*u)) & 3;
        o[k] = (scalar_t) (sc * ((nib | (hi << 4)) - 32));
    }
}


// ---------------------------------------------------------------------------------------------
// Q5_K: tiled row-block = qs[128] | qh[32] | sc[8] | mn[8] = 176 bytes; d plane: 16 x (half d, half dmin)
#define Q5T_ROWB 176
#define Q5T_TILE_BYTES (16 * Q5T_ROWB)
#define Q5T_D_BYTES 64
#define Q5T_STAGE_BYTES (Q5T_TILE_BYTES + Q5T_D_BYTES)
#define Q5T_WARPS 4
#define Q5T_STAGES 4

template <typename scalar_t>
static __global__ void mul_mat_q5k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int wk = threadIdx.y;
    const int row0 = blockIdx.x * 16;
    const int vec0 = blockIdx.y * 8;
    const int nv = min(8, nvecs - vec0);
    const int bpr = ncols / 256;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int nsb = (bpr - wk + Q5T_WARPS - 1) / Q5T_WARPS;

    __shared__ __align__(16) uint8_t smem[Q5T_WARPS][Q5T_STAGES][Q5T_STAGE_BYTES];
    const uint8_t * tile_base = wt + (size_t) blockIdx.x * bpr * Q5T_TILE_BYTES;
    const uint8_t * d_base = wd + (size_t) blockIdx.x * bpr * Q5T_D_BYTES;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int tokB = vec0 + min(g, nv - 1);
    const int n0 = min(2*t, nv - 1), n1 = min(2*t + 1, nv - 1);
    const block_q8_1 * yB = y + (size_t) tokB*(nrows_y/QK8_1);
    const block_q8_1 * y0 = y + (size_t) (vec0 + n0)*(nrows_y/QK8_1);
    const block_q8_1 * y1 = y + (size_t) (vec0 + n1)*(nrows_y/QK8_1);

    auto issue = [&](int n) {
        const int i = wk + n*Q5T_WARPS;
        const uint8_t * src = tile_base + (size_t) i * Q5T_TILE_BYTES;
        uint8_t * dstp = smem[wk][n % Q5T_STAGES];
#pragma unroll
        for (int c = lane; c < Q5T_TILE_BYTES/16; c += 32) {
            cp_async16(dstp + c*16, src + c*16);
        }
        if (lane < Q5T_D_BYTES/16) {
            cp_async16(dstp + Q5T_TILE_BYTES + lane*16, d_base + (size_t) i * Q5T_D_BYTES + lane*16);
        }
    };

#pragma unroll
    for (int s = 0; s < Q5T_STAGES - 1; ++s) {
        if (s < nsb) {
            issue(s);
        }
        cp_async_commit();
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < nsb; ++n) {
        cp_async_wait<Q5T_STAGES - 2>();
        __syncwarp();
        if (n + Q5T_STAGES - 1 < nsb) {
            issue(n + Q5T_STAGES - 1);
        }
        cp_async_commit();

        const int i = wk + n*Q5T_WARPS;
        const uint8_t * stage = smem[wk][n % Q5T_STAGES];
        const uint8_t * bufA = stage + g * Q5T_ROWB;
        const uint8_t * bufB = bufA + 8 * Q5T_ROWB;
        const uint32_t * qsA = (const uint32_t *) bufA;
        const uint32_t * qsB = (const uint32_t *) bufB;
        const uint32_t * qhA = (const uint32_t *) (bufA + 128);   // qh[32]
        const uint32_t * qhB = (const uint32_t *) (bufB + 128);
        const uint32_t * scA = (const uint32_t *) (bufA + 160);   // sc[8] | mn[8]
        const uint32_t * scB = (const uint32_t *) (bufB + 160);
        const half2 * dm = (const half2 *) (stage + Q5T_TILE_BYTES);
        const float2 dmA = __half22float2(dm[g]);
        const float2 dmB = __half22float2(dm[g + 8]);
        const uint32_t sA4[4] = {scA[0], scA[1], scA[2], scA[3]};
        const uint32_t sB4[4] = {scB[0], scB[1], scB[2], scB[3]};
        // qh bit for sub-block j is bit j of every byte of qh (byte l = weight position l within the 32-block)
        const uint32_t hA0 = qhA[t], hA1 = qhA[4 + t];
        const uint32_t hB0 = qhB[t], hB1 = qhB[4 + t];

        float sbm[4] = {0.f, 0.f, 0.f, 0.f};
        float sbn[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t wA0 = qsA[8*p + t], wA1 = qsA[8*p + 4 + t];
            const uint32_t wB0 = qsB[8*p + t], wB1 = qsB[8*p + 4 + t];
#pragma unroll
            for (int hl = 0; hl < 2; ++hl) {
                const int j = 2*p + hl;
                const uint32_t mask = 0x0F0F0F0F;
                const uint32_t lA0 = hl ? ((wA0 >> 4) & mask) : (wA0 & mask);
                const uint32_t lB0 = hl ? ((wB0 >> 4) & mask) : (wB0 & mask);
                const uint32_t lA1 = hl ? ((wA1 >> 4) & mask) : (wA1 & mask);
                const uint32_t lB1 = hl ? ((wB1 >> 4) & mask) : (wB1 & mask);
                // 5th bit: bit j of each byte -> bit 4
                const int a0 = (int) (lA0 | (((hA0 >> j) << 4) & 0x10101010));
                const int a1 = (int) (lB0 | (((hB0 >> j) << 4) & 0x10101010));
                const int a2 = (int) (lA1 | (((hA1 >> j) << 4) & 0x10101010));
                const int a3 = (int) (lB1 | (((hB1 >> j) << 4) & 0x10101010));
                const int * q8 = (const int *) yB[i*8 + j].qs;
                int c[4] = {0, 0, 0, 0};
                mma_s8_16x8x32_t(c, a0, a1, a2, a3, q8[t], q8[4 + t]);
                int cs[4] = {0, 0, 0, 0};
                mma_s8_16x8x32_t(cs, 0x01010101, 0x01010101, 0x01010101, 0x01010101, q8[t], q8[4 + t]);
                const float dy0 = __low2float(y0[i*8 + j].ds);
                const float dy1 = __low2float(y1[i*8 + j].ds);
                const float sy0 = dy0 * (float) cs[0];
                const float sy1 = dy1 * (float) cs[1];
                const int scAj = (sA4[j >> 2] >> (8*(j & 3))) & 0xFF;
                const int mnAj = (sA4[2 + (j >> 2)] >> (8*(j & 3))) & 0xFF;
                const int scBj = (sB4[j >> 2] >> (8*(j & 3))) & 0xFF;
                const int mnBj = (sB4[2 + (j >> 2)] >> (8*(j & 3))) & 0xFF;
                sbm[0] += (float) (c[0]*scAj) * dy0;
                sbm[1] += (float) (c[1]*scAj) * dy1;
                sbm[2] += (float) (c[2]*scBj) * dy0;
                sbm[3] += (float) (c[3]*scBj) * dy1;
                sbn[0] += (float) mnAj * sy0;
                sbn[1] += (float) mnAj * sy1;
                sbn[2] += (float) mnBj * sy0;
                sbn[3] += (float) mnBj * sy1;
            }
        }
        acc[0] += dmA.x*sbm[0] - dmA.y*sbn[0];
        acc[1] += dmA.x*sbm[1] - dmA.y*sbn[1];
        acc[2] += dmB.x*sbm[2] - dmB.y*sbn[2];
        acc[3] += dmB.x*sbm[3] - dmB.y*sbn[3];
    }
    cp_async_wait<0>();
    __syncthreads();

    float (*red)[32][4] = (float (*)[32][4]) &smem[0][0][0];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[wk][lane][k] = acc[k];
    }
    __syncthreads();
    if (wk != 0) {
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int w = 1; w < Q5T_WARPS; ++w) {
            acc[k] += red[w][lane][k];
        }
    }
    const int r0 = row0 + g, r1 = row0 + g + 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int tk = 2*t + u;
        if (tk < nv) {
            if (r0 < nrows) dst[(size_t)(vec0 + tk)*nrows + r0] = acc[u];
            if (r1 < nrows) dst[(size_t)(vec0 + tk)*nrows + r1] = acc[2 + u];
        }
    }
}

template <typename scalar_t>
static __global__ void dequantize_q5k_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const int tid = threadIdx.x;               // 0..127: row = tid / 8, sub-block j = tid % 8
    const int r = tid >> 3;
    const int j = tid & 7;
    const int row = tile * 16 + r;
    if (row >= rows) {
        return;
    }
    const uint8_t * blk = wt + ((size_t) tile * bpr + i) * Q5T_TILE_BYTES + r * Q5T_ROWB;
    const half2 dm = ((const half2 *) (wd + ((size_t) tile * bpr + i) * Q5T_D_BYTES))[r];
    const float d = __low2float(dm) * blk[160 + j];
    const float m = __high2float(dm) * blk[168 + j];
    const int pr = j >> 1;
    scalar_t * o = out + (size_t) row * cols + i * 256 + 32 * j;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int b = blk[32*pr + k];
        const int q = ((j & 1) ? (b >> 4) : (b & 15)) | (((blk[128 + k] >> j) & 1) << 4);
        o[k] = (scalar_t) (d * q - m);
    }
}


// ---------------------------------------------------------------------------------------------
// IQ4_XS: tiled row-block = qs[128] | sc int8[8] | pad[8] = 144 bytes; d plane: 16 x half
#define I4T_ROWB 144
#define I4T_TILE_BYTES (16 * I4T_ROWB)
#define I4T_D_BYTES 32
#define I4T_STAGE_BYTES (I4T_TILE_BYTES + I4T_D_BYTES)
#define I4T_WARPS 4
#define I4T_STAGES 5

template <typename scalar_t>
static __global__ void mul_mat_iq4xs_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int wk = threadIdx.y;
    const int row0 = blockIdx.x * 16;
    const int vec0 = blockIdx.y * 8;
    const int nv = min(8, nvecs - vec0);
    const int bpr = ncols / 256;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int nsb = (bpr - wk + I4T_WARPS - 1) / I4T_WARPS;

    __shared__ __align__(16) uint8_t smem[I4T_WARPS][I4T_STAGES][I4T_STAGE_BYTES];
    const uint8_t * tile_base = wt + (size_t) blockIdx.x * bpr * I4T_TILE_BYTES;
    const uint8_t * d_base = wd + (size_t) blockIdx.x * bpr * I4T_D_BYTES;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int tokB = vec0 + min(g, nv - 1);
    const int n0 = min(2*t, nv - 1), n1 = min(2*t + 1, nv - 1);
    const block_q8_1 * yB = y + (size_t) tokB*(nrows_y/QK8_1);
    const block_q8_1 * y0 = y + (size_t) (vec0 + n0)*(nrows_y/QK8_1);
    const block_q8_1 * y1 = y + (size_t) (vec0 + n1)*(nrows_y/QK8_1);
    __shared__ uint8_t values[16];
    if (wk == 0 && lane < 16) {
        values[lane] = (uint8_t) kvalues_iq4nl[lane];
    }
    __syncthreads();

    auto issue = [&](int n) {
        const int i = wk + n*I4T_WARPS;
        const uint8_t * src = tile_base + (size_t) i * I4T_TILE_BYTES;
        uint8_t * dstp = smem[wk][n % I4T_STAGES];
#pragma unroll
        for (int c = lane; c < I4T_TILE_BYTES/16; c += 32) {
            cp_async16(dstp + c*16, src + c*16);
        }
        if (lane < I4T_D_BYTES/16) {
            cp_async16(dstp + I4T_TILE_BYTES + lane*16, d_base + (size_t) i * I4T_D_BYTES + lane*16);
        }
    };

#pragma unroll
    for (int s = 0; s < I4T_STAGES - 1; ++s) {
        if (s < nsb) {
            issue(s);
        }
        cp_async_commit();
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < nsb; ++n) {
        cp_async_wait<I4T_STAGES - 2>();
        __syncwarp();
        if (n + I4T_STAGES - 1 < nsb) {
            issue(n + I4T_STAGES - 1);
        }
        cp_async_commit();

        const int i = wk + n*I4T_WARPS;
        const uint8_t * stage = smem[wk][n % I4T_STAGES];
        const uint8_t * bufA = stage + g * I4T_ROWB;
        const uint8_t * bufB = bufA + 8 * I4T_ROWB;
        const uint32_t * qsA = (const uint32_t *) bufA;
        const uint32_t * qsB = (const uint32_t *) bufB;
        const uint32_t * scA = (const uint32_t *) (bufA + 128);   // int8 scales[8]
        const uint32_t * scB = (const uint32_t *) (bufB + 128);
        const half * dh = (const half *) (stage + I4T_TILE_BYTES);
        const float dA = __half2float(dh[g]);
        const float dB = __half2float(dh[g + 8]);
        const uint32_t sA2[2] = {scA[0], scA[1]};
        const uint32_t sB2[2] = {scB[0], scB[1]};

        float sb[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int a0, a1, a2, a3;
            get_int_from_table_16(qsA[4*j + t], values, a0, a2);   // low nibbles -> k 4t.., high nibbles -> k 16+4t..
            get_int_from_table_16(qsB[4*j + t], values, a1, a3);
            const int * q8 = (const int *) yB[i*8 + j].qs;
            int c[4] = {0, 0, 0, 0};
            mma_s8_16x8x32_t(c, a0, a1, a2, a3, q8[t], q8[4 + t]);
            const float dy0 = __low2float(y0[i*8 + j].ds);
            const float dy1 = __low2float(y1[i*8 + j].ds);
            const int scAj = ((int) (sA2[j >> 2] << (24 - 8*(j & 3)))) >> 24;
            const int scBj = ((int) (sB2[j >> 2] << (24 - 8*(j & 3)))) >> 24;
            sb[0] += (float) (c[0]*scAj) * dy0;
            sb[1] += (float) (c[1]*scAj) * dy1;
            sb[2] += (float) (c[2]*scBj) * dy0;
            sb[3] += (float) (c[3]*scBj) * dy1;
        }
        acc[0] += sb[0] * dA;
        acc[1] += sb[1] * dA;
        acc[2] += sb[2] * dB;
        acc[3] += sb[3] * dB;
    }
    cp_async_wait<0>();
    __syncthreads();

    float (*red)[32][4] = (float (*)[32][4]) &smem[0][0][0];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[wk][lane][k] = acc[k];
    }
    __syncthreads();
    if (wk != 0) {
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int w = 1; w < I4T_WARPS; ++w) {
            acc[k] += red[w][lane][k];
        }
    }
    const int r0 = row0 + g, r1 = row0 + g + 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int tk = 2*t + u;
        if (tk < nv) {
            if (r0 < nrows) dst[(size_t)(vec0 + tk)*nrows + r0] = acc[u];
            if (r1 < nrows) dst[(size_t)(vec0 + tk)*nrows + r1] = acc[2 + u];
        }
    }
}

template <typename scalar_t>
static __global__ void dequantize_iq4xs_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const int tid = threadIdx.x;               // 0..127: row = tid / 8, sub-block j = tid % 8
    const int r = tid >> 3;
    const int j = tid & 7;
    const int row = tile * 16 + r;
    if (row >= rows) {
        return;
    }
    const uint8_t * blk = wt + ((size_t) tile * bpr + i) * I4T_TILE_BYTES + r * I4T_ROWB;
    const float d = __half2float(((const half *) (wd + ((size_t) tile * bpr + i) * I4T_D_BYTES))[r]) * ((const int8_t *) (blk + 128))[j];
    const int8_t * vals = (const int8_t *) kvalues_iq4nl;
    scalar_t * o = out + (size_t) row * cols + i * 256 + 32 * j;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int b = blk[16*j + k];
        o[k] = (scalar_t) (d * vals[b & 15]);
        o[k + 16] = (scalar_t) (d * vals[b >> 4]);
    }
}

// ---------------------------------------------------------------------------------------------
// IQ3_XXS: tiled row-block = q3[64] | gas[32] = 96 bytes (as in the GGUF block, without d); d plane: 16 x half
#define I3T_ROWB 96
#define I3T_TILE_BYTES (16 * I3T_ROWB)
#define I3T_D_BYTES 32
#define I3T_STAGE_BYTES (I3T_TILE_BYTES + I3T_D_BYTES)
#define I3T_WARPS 4
#define I3T_STAGES 6

template <typename scalar_t>
static __global__ void mul_mat_iq3xxs_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int wk = threadIdx.y;
    const int row0 = blockIdx.x * 16;
    const int vec0 = blockIdx.y * 8;
    const int nv = min(8, nvecs - vec0);
    const int bpr = ncols / 256;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int nsb = (bpr - wk + I3T_WARPS - 1) / I3T_WARPS;

    __shared__ __align__(16) uint8_t smem[I3T_WARPS][I3T_STAGES][I3T_STAGE_BYTES];
    const uint8_t * tile_base = wt + (size_t) blockIdx.x * bpr * I3T_TILE_BYTES;
    const uint8_t * d_base = wd + (size_t) blockIdx.x * bpr * I3T_D_BYTES;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int tokB = vec0 + min(g, nv - 1);
    const int n0 = min(2*t, nv - 1), n1 = min(2*t + 1, nv - 1);
    const block_q8_1 * yB = y + (size_t) tokB*(nrows_y/QK8_1);
    const block_q8_1 * y0 = y + (size_t) (vec0 + n0)*(nrows_y/QK8_1);
    const block_q8_1 * y1 = y + (size_t) (vec0 + n1)*(nrows_y/QK8_1);

    __shared__ uint32_t sgrid[256];
    __shared__ uint64_t ssig[128];
    for (int k = wk*32 + lane; k < 256; k += I3T_WARPS*32) {
        sgrid[k] = iq3xxs_grid[k];
        if (k < 128) {
            ssig[k] = ksigns64[k];
        }
    }
    __syncthreads();

    auto issue = [&](int n) {
        const int i = wk + n*I3T_WARPS;
        const uint8_t * src = tile_base + (size_t) i * I3T_TILE_BYTES;
        uint8_t * dstp = smem[wk][n % I3T_STAGES];
#pragma unroll
        for (int c = lane; c < I3T_TILE_BYTES/16; c += 32) {
            cp_async16(dstp + c*16, src + c*16);
        }
        if (lane < I3T_D_BYTES/16) {
            cp_async16(dstp + I3T_TILE_BYTES + lane*16, d_base + (size_t) i * I3T_D_BYTES + lane*16);
        }
    };

#pragma unroll
    for (int s = 0; s < I3T_STAGES - 1; ++s) {
        if (s < nsb) {
            issue(s);
        }
        cp_async_commit();
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int n = 0; n < nsb; ++n) {
        cp_async_wait<I3T_STAGES - 2>();
        __syncwarp();
        if (n + I3T_STAGES - 1 < nsb) {
            issue(n + I3T_STAGES - 1);
        }
        cp_async_commit();

        const int i = wk + n*I3T_WARPS;
        const uint8_t * stage = smem[wk][n % I3T_STAGES];
        const uint8_t * bufA = stage + g * I3T_ROWB;
        const uint8_t * bufB = bufA + 8 * I3T_ROWB;
        const uint32_t * gasA = (const uint32_t *) (bufA + 64);
        const uint32_t * gasB = (const uint32_t *) (bufB + 64);
        const half * dh = (const half *) (stage + I3T_TILE_BYTES);
        const float dA = __half2float(dh[g]) * 0.25f;
        const float dB = __half2float(dh[g + 8]) * 0.25f;

        float sb[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int j = 0; j < 8; ++j) {                 // 32-weight sub-block
            int a[4];
            int fAB[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint8_t * buf = r == 0 ? bufA : bufB;
                const uint32_t aux32 = (r == 0 ? gasA : gasB)[j];
                const uint8_t * q3 = buf + 8*j;
#pragma unroll
                for (int h = 0; h < 2; ++h) {           // words t (k 4t..) and 4+t (k 16+4t..)
                    const int w = h*4 + t;
                    const int l = w >> 1;
                    const uint32_t grid = sgrid[q3[w]];
                    const uint32_t * signs = (const uint32_t *) (ssig + ((aux32 >> (7*l)) & 127));
                    const uint32_t sg = signs[w & 1];
                    a[h*2 + r] = __vsub4(grid ^ sg, sg);
                }
                fAB[r] = 1 + 2*(int) (aux32 >> 28);   // scale (0.5 + s) * 0.5 = 0.25 * (1 + 2 s)
            }
            const int * q8 = (const int *) yB[i*8 + j].qs;
            int c[4] = {0, 0, 0, 0};
            mma_s8_16x8x32_t(c, a[0], a[1], a[2], a[3], q8[t], q8[4 + t]);
            const float dy0 = __low2float(y0[i*8 + j].ds);
            const float dy1 = __low2float(y1[i*8 + j].ds);
            sb[0] += (float) (c[0]*fAB[0]) * dy0;
            sb[1] += (float) (c[1]*fAB[0]) * dy1;
            sb[2] += (float) (c[2]*fAB[1]) * dy0;
            sb[3] += (float) (c[3]*fAB[1]) * dy1;
        }
        acc[0] += sb[0] * dA;
        acc[1] += sb[1] * dA;
        acc[2] += sb[2] * dB;
        acc[3] += sb[3] * dB;
    }
    cp_async_wait<0>();
    __syncthreads();

    float (*red)[32][4] = (float (*)[32][4]) &smem[0][0][0];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[wk][lane][k] = acc[k];
    }
    __syncthreads();
    if (wk != 0) {
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int w = 1; w < I3T_WARPS; ++w) {
            acc[k] += red[w][lane][k];
        }
    }
    const int r0 = row0 + g, r1 = row0 + g + 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int tk = 2*t + u;
        if (tk < nv) {
            if (r0 < nrows) dst[(size_t)(vec0 + tk)*nrows + r0] = acc[u];
            if (r1 < nrows) dst[(size_t)(vec0 + tk)*nrows + r1] = acc[2 + u];
        }
    }
}

template <typename scalar_t>
static __global__ void dequantize_iq3xxs_tiled(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const int tid = threadIdx.x;               // 0..127: row = tid / 8, sub-block j = tid % 8
    const int r = tid >> 3;
    const int j = tid & 7;
    const int row = tile * 16 + r;
    if (row >= rows) {
        return;
    }
    const uint8_t * blk = wt + ((size_t) tile * bpr + i) * I3T_TILE_BYTES + r * I3T_ROWB;
    const float dd = __half2float(((const half *) (wd + ((size_t) tile * bpr + i) * I3T_D_BYTES))[r]);
    const uint32_t aux32 = ((const uint32_t *) (blk + 64))[j];
    const float d = dd * 0.25f * (1 + 2 * (int) (aux32 >> 28));
    const uint8_t * q3 = blk + 8*j;
    scalar_t * o = out + (size_t) row * cols + i * 256 + 32 * j;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t grid = iq3xxs_grid[q3[w]];
        const uint32_t * signs = (const uint32_t *) (ksigns64 + ((aux32 >> (7*(w >> 1))) & 127));
        const uint32_t sg = signs[w & 1];
        const int v = __vsub4(grid ^ sg, sg);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            o[4*w + b] = (scalar_t) (d * (float) ((int8_t) ((v >> (8*b)) & 0xFF)));
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Fast dequantization of the tiled layouts: one block per (tile, superblock) stages the tile in shared
// memory with 16-byte loads, then thread c writes element c of every row (coalesced 2-byte stores).
struct DqQ3K {
    static constexpr int ROWB = Q3T_ROWB, TILE = Q3T_TILE_BYTES, DB = Q3T_D_BYTES;
    static __device__ __forceinline__ float val(const uint8_t * blk, const uint8_t * dpl, const int r, const int c) {
        const int s = c >> 4, k = c & 15;
        const int grp = s >> 1, half_ = grp >> 2, j = grp & 3;
        const int pos = (s & 1) * 16 + k;
        const int q = (blk[32 + half_ * 32 + pos] >> (2 * j)) & 3;
        const int h = (blk[pos] >> grp) & 1;
        const float d = __half2float(((const half *) dpl)[r]);
        return d * ((const int8_t *) (blk + 96))[s] * (q - (h ? 0 : 4));
    }
};
struct DqQ4K {
    static constexpr int ROWB = Q4T_ROWB, TILE = Q4T_TILE_BYTES, DB = Q4T_D_BYTES;
    static __device__ __forceinline__ float val(const uint8_t * blk, const uint8_t * dpl, const int r, const int c) {
        const int j = c >> 5, k = c & 31;
        const int b = blk[32 * (j >> 1) + k];
        const int q = (j & 1) ? (b >> 4) : (b & 15);
        const half2 dm = ((const half2 *) dpl)[r];
        return __low2float(dm) * blk[128 + j] * q - __high2float(dm) * blk[136 + j];
    }
};
struct DqQ5K {
    static constexpr int ROWB = Q5T_ROWB, TILE = Q5T_TILE_BYTES, DB = Q5T_D_BYTES;
    static __device__ __forceinline__ float val(const uint8_t * blk, const uint8_t * dpl, const int r, const int c) {
        const int j = c >> 5, k = c & 31;
        const int b = blk[32 * (j >> 1) + k];
        const int q = ((j & 1) ? (b >> 4) : (b & 15)) | (((blk[128 + k] >> j) & 1) << 4);
        const half2 dm = ((const half2 *) dpl)[r];
        return __low2float(dm) * blk[160 + j] * q - __high2float(dm) * blk[168 + j];
    }
};
struct DqQ6K {
    static constexpr int ROWB = Q6T_ROWB, TILE = Q6T_TILE_BYTES, DB = Q6T_D_BYTES;
    static __device__ __forceinline__ float val(const uint8_t * blk, const uint8_t * dpl, const int r, const int c) {
        const int s = c >> 4;
        const int h = s >> 3, lh = s & 1, u = (s >> 1) & 3;
        const int l = 16 * lh + (c & 15);
        const int qlb = blk[h * 64 + (u & 1) * 32 + l];
        const int nib = (u < 2) ? (qlb & 15) : (qlb >> 4);
        const int hi = (blk[128 + h * 32 + l] >> (2 * u)) & 3;
        const float d = __half2float(((const half *) dpl)[r]);
        return d * ((const int8_t *) (blk + 192))[s] * ((nib | (hi << 4)) - 32);
    }
};
struct DqIQ3XXS {
    static constexpr int ROWB = I3T_ROWB, TILE = I3T_TILE_BYTES, DB = I3T_D_BYTES;
    static __device__ __forceinline__ float val(const uint8_t * blk, const uint8_t * dpl, const int r, const int c) {
        const int j = c >> 5, w = (c & 31) >> 2, b4 = c & 3;
        const uint32_t aux32 = ((const uint32_t *) (blk + 64))[j];
        const uint32_t grid = iq3xxs_grid[blk[8 * j + w]];
        const uint32_t * signs = (const uint32_t *) (ksigns64 + ((aux32 >> (7 * (w >> 1))) & 127));
        const uint32_t sg = signs[w & 1];
        const int v = __vsub4(grid ^ sg, sg);
        const float d = __half2float(((const half *) dpl)[r]) * 0.25f * (1 + 2 * (int) (aux32 >> 28));
        return d * (float) ((int8_t) ((v >> (8 * b4)) & 0xFF));
    }
};

template <typename scalar_t, typename Fmt>
static __global__ void dequantize_tiled_fast(const uint8_t * __restrict__ wt, const uint8_t * __restrict__ wd, scalar_t * __restrict__ out, const int cols, const int rows) {
    __shared__ __align__(16) uint8_t sm[Fmt::TILE + Fmt::DB];
    const int tile = blockIdx.x;
    const int i = blockIdx.y;
    const int bpr = cols / 256;
    const uint4 * src = (const uint4 *) (wt + ((size_t) tile * bpr + i) * Fmt::TILE);
    const uint4 * dsrc = (const uint4 *) (wd + ((size_t) tile * bpr + i) * Fmt::DB);
    uint4 * dst = (uint4 *) sm;
    for (int o = threadIdx.x; o < Fmt::TILE / 16; o += blockDim.x) {
        dst[o] = src[o];
    }
    if (threadIdx.x < Fmt::DB / 16) {
        dst[Fmt::TILE / 16 + threadIdx.x] = dsrc[threadIdx.x];
    }
    __syncthreads();
    const int c = threadIdx.x;      // 0..255
    const uint8_t * dpl = sm + Fmt::TILE;
#pragma unroll 4
    for (int r = 0; r < 16; ++r) {
        const int row = tile * 16 + r;
        if (row < rows) {
            out[(size_t) row * cols + i * 256 + c] = (scalar_t) Fmt::val(sm + r * Fmt::ROWB, dpl, r, c);
        }
    }
}
