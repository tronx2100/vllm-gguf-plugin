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
