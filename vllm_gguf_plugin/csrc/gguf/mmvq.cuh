// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
// Each block handles up to MMVQ_VECS input vectors at once, so every weight block is read
// from memory once per block instead of once per vector (speculative decoding verifies several
// tokens per step; one block per vector made that cost scale linearly with the token count).
#define MMVQ_VECS 8

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const int vec0 = blockIdx.y*MMVQ_VECS;

    if (row >= nrows || vec0 >= nvecs) {
        return;
    }
    const int nv = min(MMVQ_VECS, nvecs - vec0);

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    // partial sum per thread and vector
    float tmp[MMVQ_VECS];
#pragma unroll
    for (int v = 0; v < MMVQ_VECS; ++v) {
        tmp[v] = 0.0f;
    }

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index
        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

#pragma unroll
        for (int v = 0; v < MMVQ_VECS; ++v) {
            if (v < nv) {
                const int iby = (vec0 + v)*(nrows_y/QK8_1) + i * (qk/QK8_1); // y block index that aligns with ibx
                tmp[v] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
            }
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int v = 0; v < MMVQ_VECS; ++v) {
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            tmp[v] += VLLM_SHFL_XOR_SYNC(tmp[v], mask);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int v = 0; v < MMVQ_VECS; ++v) {
            if (v < nv) {
                dst[(vec0 + v)*nrows + row] = tmp[v];
            }
        }
    }
}


// IQ3_XXS: decode the grid/sign values of a weight block once and reuse them for all vectors.
template <typename scalar_t, int NV>
static __global__ void mul_mat_vec_iq3_xxs_multi(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const int vec0 = blockIdx.y*NV;

    if (row >= nrows || vec0 >= nvecs) {
        return;
    }
    const int nv = min(NV, nvecs - vec0);

    constexpr int qk = QK_K;
    constexpr int qi = QI3_XXS;
    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = 1 * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        tmp[v] = 0.0f;
    }

    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / qi; i < blocks_per_row; i += blocks_per_warp) {
        const block_iq3_xxs * bq2 = &x[row*blocks_per_row + i];
        const int ib32 = threadIdx.x % qi;

        const uint8_t  * q3 = bq2->qs + 8*ib32;
        const uint16_t * gas = (const uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
        uint32_t aux32 = gas[0] | (gas[1] << 16);
        int grid_l[4], grid_h[4];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t * grid1 = iq3xxs_grid + q3[2*l+0];
            const uint32_t * grid2 = iq3xxs_grid + q3[2*l+1];
            const uint32_t * signs = (const uint32_t *)(ksigns64 + (aux32 & 127));
            grid_l[l] = __vsub4(grid1[0] ^ signs[0], signs[0]);
            grid_h[l] = __vsub4(grid2[0] ^ signs[1], signs[1]);
            aux32 >>= 7;
        }
        const float dx = __half2float(bq2->d) * (0.5f + aux32) * 0.5f;

#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (v < nv) {
                const block_q8_1 * b8 = &y[(vec0 + v)*(nrows_y/QK8_1) + i * (qk/QK8_1)] + ib32;
                const int * q8 = (const int *) b8->qs;
                int sumi = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    sumi = __dp4a(grid_l[l], q8[2*l+0], sumi);
                    sumi = __dp4a(grid_h[l], q8[2*l+1], sumi);
                }
                tmp[v] += dx * __low2float(b8->ds) * sumi;
            }
        }
    }

#pragma unroll
    for (int v = 0; v < NV; ++v) {
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            tmp[v] += VLLM_SHFL_XOR_SYNC(tmp[v], mask);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (v < nv) {
                dst[(vec0 + v)*nrows + row] = tmp[v];
            }
        }
    }
}


// IQ4_XS: look up the 4-bit values of a weight block once and reuse them for all vectors.
template <typename scalar_t, int NV>
static __global__ void mul_mat_vec_iq4_xs_multi(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const int vec0 = blockIdx.y*NV;

    if (row >= nrows || vec0 >= nvecs) {
        return;
    }
    const int nv = min(NV, nvecs - vec0);

    constexpr int qk = QK_K;
    constexpr int qi = QI4_XS;
    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = 1 * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        tmp[v] = 0.0f;
    }

    const block_iq4_xs * x = (const block_iq4_xs *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    for (auto i = threadIdx.x / qi; i < blocks_per_row; i += blocks_per_warp) {
        const block_iq4_xs * bq4 = &x[row*blocks_per_row + i];
        const int ib32 = threadIdx.x % qi;

        const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
        const int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
        const float dx = __half2float(bq4->d) * (ls - 32);
        int v1[4], v2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            get_int_from_table_16(q4[j], values, v1[j], v2[j]);
        }

#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (v < nv) {
                const block_q8_1 * b8 = &y[(vec0 + v)*(nrows_y/QK8_1) + i * (qk/QK8_1)] + ib32;
                const int * q8 = (const int *) b8->qs;
                int sumi1 = 0, sumi2 = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi1 = __dp4a(v1[j], q8[j+0], sumi1);
                    sumi2 = __dp4a(v2[j], q8[j+4], sumi2);
                }
                tmp[v] += dx * __low2float(b8->ds) * (sumi1 + sumi2);
            }
        }
    }

#pragma unroll
    for (int v = 0; v < NV; ++v) {
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            tmp[v] += VLLM_SHFL_XOR_SYNC(tmp[v], mask);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (v < nv) {
                dst[(vec0 + v)*nrows + row] = tmp[v];
            }
        }
    }
}


// Generic multi-vector kernel for formats whose per-thread weight decode is split into
// Op::decode (weights only, done once) and Op::dot (per input vector).
template <typename scalar_t, int NV, int qk, int qi, typename block_q_t, int vdr, typename Op>
static __global__ void mul_mat_vec_q_op_multi(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const int vec0 = blockIdx.y*NV;

    if (row >= nrows || vec0 >= nvecs) {
        return;
    }
    const int nv = min(NV, nvecs - vec0);

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        tmp[v] = 0.0f;
    }

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int iqs = vdr * (threadIdx.x % (qi/vdr));
        typename Op::State st;
        Op::decode(&x[row*blocks_per_row + i], iqs, st);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (v < nv) {
                tmp[v] += Op::dot(st, &y[(vec0 + v)*(nrows_y/QK8_1) + i * (qk/QK8_1)], iqs);
            }
        }
    }

#pragma unroll
    for (int v = 0; v < NV; ++v) {
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            tmp[v] += VLLM_SHFL_XOR_SYNC(tmp[v], mask);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (v < nv) {
                dst[(vec0 + v)*nrows + row] = tmp[v];
            }
        }
    }
}

struct Q6KOp {
    struct State { int vi[QR6_K]; int sc[QR6_K]; float d; int bq8_offset; };
    static __device__ __forceinline__ void decode(const block_q6_K * bq6_K, const int iqs, State & st) {
        st.bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
        const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
        const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
        const int vl = get_int_from_uint8(bq6_K->ql, iqs);
        const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
        const int8_t * scales = bq6_K->scales + scale_offset;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
            st.vi[i] = __vsubss4((vil | vih), 0x20202020);
            st.sc[i] = scales[4*i];
        }
        st.d = __half2float(bq6_K->d);
    }
    static __device__ __forceinline__ float dot(const State & st, const block_q8_1 * bq8_1, const int iqs) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const int u = get_int_from_int8_aligned(bq8_1[st.bq8_offset + 2*i].qs, iqs % QI8_1);
            const float d8 = __low2float(bq8_1[st.bq8_offset + 2*i].ds);
            sumf += d8 * (__dp4a(st.vi[i], u, 0) * st.sc[i]);
        }
        return st.d * sumf;
    }
};

struct Q5KOp {
    struct State { int v0[QR5_K]; int v1[QR5_K]; uint8_t sc[QR5_K]; uint8_t m[QR5_K]; half2 dm; int bq8_offset; };
    static __device__ __forceinline__ void decode(const block_q5_K * bq5_K, const int iqs, State & st) {
        st.bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
        const int * ql = (const int *)(bq5_K->qs + 16 * st.bq8_offset + 4 * ((iqs/2)%4));
        const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));
        const int vl0 = ql[0], vl1 = ql[4];
        const int vh0 = qh[0] >> st.bq8_offset, vh1 = qh[4] >> st.bq8_offset;
        const uint16_t * scales = (const uint16_t *)bq5_K->scales;
        uint16_t aux[2];
        const int j = st.bq8_offset/2;
        if (j < 2) {
            aux[0] = scales[j+0] & 0x3f3f;
            aux[1] = scales[j+2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
        }
        const uint8_t * sc = (const uint8_t *)aux;
        const uint8_t * m  = sc + 2;
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            st.v0[i] = ((vl0 >> (4*i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
            st.v1[i] = ((vl1 >> (4*i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
            st.sc[i] = sc[i];
            st.m[i] = m[i];
        }
        st.dm = bq5_K->dm;
    }
    static __device__ __forceinline__ float dot(const State & st, const block_q8_1 * bq8_1, const int iqs) {
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1 * bq8i = bq8_1 + st.bq8_offset + i;
            const float d8 = __low2float(bq8i->ds);
            const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
            const int u0 = q8[0], u1 = q8[4];
            const int dot1 = __dp4a(st.v0[i], u0, __dp4a(st.v1[i], u1, 0));
            const int dot2 = __dp4a(0x01010101, u0, __dp4a(0x01010101, u1, 0));
            sumf_d += d8 * (dot1 * st.sc[i]);
            sumf_m += d8 * (dot2 * st.m[i]);
        }
        const float2 dm5f = __half22float2(st.dm);
        return dm5f.x*sumf_d - dm5f.y*sumf_m;
    }
};

template<typename scalar_t, int NV, int qk, int qi, typename block_q_t, int vdr, typename Op>
static void mul_mat_vec_q_op_launch(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + NV - 1) / NV, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_op_multi<scalar_t, NV, qk, qi, block_q_t, vdr, Op>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t, int qk, int qi, typename block_q_t, int vdr, typename Op>
static void mul_mat_vec_q_op_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    if (nvecs == 1) {
        mul_mat_vec_q_op_launch<scalar_t, 1, qk, qi, block_q_t, vdr, Op>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else if (nvecs == 2) {
        mul_mat_vec_q_op_launch<scalar_t, 2, qk, qi, block_q_t, vdr, Op>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else if (nvecs <= 4) {
        mul_mat_vec_q_op_launch<scalar_t, 4, qk, qi, block_q_t, vdr, Op>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else {
        mul_mat_vec_q_op_launch<scalar_t, 8, qk, qi, block_q_t, vdr, Op>(vx, vy, dst, ncols, nrows, nvecs, stream);
    }
}

template<typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mul_mat_vec_q_op_cuda<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, Q5KOp>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mul_mat_vec_q_op_cuda<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, Q6KOp>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t, int NV>
static void mul_mat_vec_iq3_xxs_launch(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + NV - 1) / NV, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_iq3_xxs_multi<scalar_t, NV>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    if (nvecs == 1) {
        mul_mat_vec_iq3_xxs_launch<scalar_t, 1>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else if (nvecs == 2) {
        mul_mat_vec_iq3_xxs_launch<scalar_t, 2>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else if (nvecs <= 4) {
        mul_mat_vec_iq3_xxs_launch<scalar_t, 4>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else {
        mul_mat_vec_iq3_xxs_launch<scalar_t, 8>(vx, vy, dst, ncols, nrows, nvecs, stream);
    }
}

template<typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t, int NV>
static void mul_mat_vec_iq4_xs_launch(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + NV - 1) / NV, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_iq4_xs_multi<scalar_t, NV>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    if (nvecs == 1) {
        mul_mat_vec_iq4_xs_launch<scalar_t, 1>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else if (nvecs == 2) {
        mul_mat_vec_iq4_xs_launch<scalar_t, 2>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else if (nvecs <= 4) {
        mul_mat_vec_iq4_xs_launch<scalar_t, 4>(vx, vy, dst, ncols, nrows, nvecs, stream);
    } else {
        mul_mat_vec_iq4_xs_launch<scalar_t, 8>(vx, vy, dst, ncols, nrows, nvecs, stream);
    }
}

template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, (nvecs + MMVQ_VECS - 1) / MMVQ_VECS, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}
