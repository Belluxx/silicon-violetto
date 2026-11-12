#include <metal_stdlib>
using namespace metal;

constant constexpr uint H = 1280;      // hidden size
constant constexpr uint D = 128;       // head dim
constant constexpr uint NQ = 10;       // query heads
constant constexpr uint NKV = 2;       // kv heads
constant constexpr uint GROUP = 5;     // query heads per kv head
constant constexpr uint QKV = 1792;    // q | k | v row width
constant constexpr uint KOFF = 1280;   // k offset in a qkv row
constant constexpr uint VOFF = 1536;   // v offset in a qkv row
constant constexpr uint VE_W = 256;    // value-embedding row width (2 heads x 128)
constant constexpr float RMS_EPS = 1.1920928955078125e-07f;  // 2^-23
constant constexpr float XSA_EPS = 1e-4f;
constant constexpr uint PART = 132;    // floats per attention partial: l, 3 pad (o 16-byte aligned), o[128]

inline float bf(ushort u) { return as_type<float>(uint(u) << 16); }
inline float4 bf4(uint2 u) {
    return float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                  as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
}
// Round to nearest even, as torch's .to(bfloat16) does (inputs are finite).
inline ushort to_bf(float f) {
    uint u = as_type<uint>(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// erf for fp32 (Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7); Metal has none.
inline float erf_f(float x) {
    float a = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a);
    return copysign(y, x);
}
inline float gelu(float x) { return 0.5f * x * (1.0f + erf_f(x * 0.70710678118654752f)); }

// Sum over a threadgroup of `nsg` simdgroups. `red` needs nsg floats.
inline float tg_sum(float v, threadgroup float* red, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = 0; i < nsg; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// ------------------------------------------------------------------ embedding
// Sequence state lives on the GPU, so that a step can be encoded before the previous one
// has decided how far its sequences advance:
//   - lens[slot]: the position of the slot's next input token;
//   - feed[slot][FEED_W]: GPU-fed input tokens, the slot's next input then its drafts;
//   - tree[slot][FEED_W]: each draft's node in the tree over the next input, parents before
//     children. A chain is row i at depth i, under row i - 1.
// The CPU writes each token's (slot, row in its run) to `tin`, and embed resolves
// meta[t] = (slot, lens[slot] + depth) for the step's later kernels. A row's K/V go to the
// cache at its run's first position plus its row (cache_pos), since siblings share a
// position; kv_compact then moves the accepted path's into place.
constant constexpr uint FEED_ROW = 0xffff0000u;  // input token FEED_ROW + i: feed[slot][i]
constant constexpr uint FEED_W = 16;

// A node: depth | parent << 8 | the rows it sees << 16 (its ancestors and itself). A row
// that is never accepted has no parent.
constant constexpr uint NO_PARENT = 0xffu;
inline uint tree_node(uint depth, uint parent, uint sees) { return depth | parent << 8 | sees << 16; }
inline uint chain_node(uint row) { return tree_node(row, row == 0 ? NO_PARENT : row - 1, (2u << row) - 1u); }
inline uint dead_node(uint row) { return tree_node(1, NO_PARENT, 1u | 1u << row); }  // sees the root and itself
inline uint node_depth(uint node) { return node & 0xffu; }
inline uint node_parent(uint node) { return (node >> 8) & 0xffu; }
inline uint node_sees(uint node) { return node >> 16; }

// A token's row in the KV cache: its run's first position plus its row in the run.
inline uint cache_pos(device const uint2* meta, device const uint2* tin, uint t) {
    return meta[t - tin[t].y].y + tin[t].y;
}

// out[t] = rms_norm(embed[tokens[t]]). A GPU-fed token is resolved here, in place: its id,
// its position from its node's depth, and the keys of its run it sees (`vis`). One
// threadgroup of 256 per token.
kernel void embed(device const ushort* table [[buffer(0)]],
                  device uint* tokens [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  device const uint2* tin [[buffer(3)]],
                  device uint2* meta [[buffer(4)]],
                  device const uint* lens [[buffer(5)]],
                  device const uint* feed [[buffer(6)]],
                  device const uint* tree [[buffer(7)]],
                  device uint* vis [[buffer(8)]],
                  uint t [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    const uint slot = tin[t].x;
    uint tok = tokens[t], depth = tin[t].y;
    if (tok >= FEED_ROW) {
        const uint i = tok - FEED_ROW, node = tree[slot * FEED_W + i];
        tok = feed[slot * FEED_W + i];
        depth = node_depth(node);
        if (tid == 0) vis[t] = node_sees(node);
    }
    device const ushort* row = table + ulong(tok) * H;
    float v[5];
    float ss = 0.0f;
    for (uint i = 0; i < 5; i++) { v[i] = bf(row[tid + 256 * i]); ss += v[i] * v[i]; }
    float inv = rsqrt(tg_sum(ss, red, sg, lane, 8) / float(H) + RMS_EPS);
    if (tid == 0) {  // after tg_sum's barriers: every thread has read it
        tokens[t] = tok;
        meta[t] = uint2(slot, lens[slot] + depth);
    }
    for (uint i = 0; i < 5; i++) out[ulong(t) * H + tid + 256 * i] = v[i] * inv;
}

// out[r] = rms_norm(in[rows[r]]). One threadgroup of 256 per output row.
kernel void rmsnorm_rows(device const float* in [[buffer(0)]],
                         device const uint* rows [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         uint r [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float* x = in + ulong(rows[r]) * H;
    float v[5];
    float ss = 0.0f;
    for (uint i = 0; i < 5; i++) { v[i] = x[tid + 256 * i]; ss += v[i] * v[i]; }
    float inv = rsqrt(tg_sum(ss, red, sg, lane, 8) / float(H) + RMS_EPS);
    for (uint i = 0; i < 5; i++) out[ulong(r) * H + tid + 256 * i] = v[i] * inv;
}

// out[t] = bf16(rms_norm(in[t])), kept as float: a layer's output, tapped for the drafter.
// One threadgroup of 256 per token.
kernel void tap_rows(device const float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     uint t [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float* x = in + ulong(t) * H;
    float v[5];
    float ss = 0.0f;
    for (uint i = 0; i < 5; i++) { v[i] = x[tid + 256 * i]; ss += v[i] * v[i]; }
    float inv = rsqrt(tg_sum(ss, red, sg, lane, 8) / float(H) + RMS_EPS);
    for (uint i = 0; i < 5; i++) out[ulong(t) * H + tid + 256 * i] = bf(to_bf(v[i] * inv));
}

// ----------------------------------------------------------------- matmuls
// Y[t][row] = epilogue(W[row] . X[t]); W [m][k] row-major, X fp32 [n][k]. W is bf16, or int8
// with one bf16 scale per group of 2^gshift consecutive k in S [m][k >> gshift]. A format
// (Bf16W, I8W) loads 8 consecutive weights of a row at a time.
enum : uint { EPI_NORM = 1, EPI_RESID = 2, EPI_SWIGLU = 3, EPI_HEAD = 4, EPI_RAW = 5 };
enum : uint { FLAG_WRITE_INV = 1, FLAG_WRITE_RAW = 2 };

struct MatArgs {
    uint n, k, m;       // tokens, inputs, outputs
    uint epi, flags;
    uint vocab_real;    // EPI_HEAD: rows >= this are masked
    float scale;        // matrix scale: qkv_scale or o_scale, 1 for the other matrices
    float ra, rb;       // EPI_RESID: y = ra * base + rb * (scale * acc)
    uint ksplit, kchunk; // GEMM only: k ranges of kchunk, partials summed by gemm_splitk_reduce
    uint gshift;        // int8: log2 of the scale group
};

// The scale is a compile-time 1 for bf16, so its multiplies fold away.
struct Bf16W {
    typedef uint4 chunk;
    static chunk load(device const uchar* w, ulong e) { return *(device const uint4*)(w + 2 * e); }
    static float scale(device const ushort* s, ulong i) { return 1.0f; }
    static float4 lo(chunk c) { return bf4(c.xy); }
    static float4 hi(chunk c) { return bf4(c.zw); }
};
struct I8W {
    typedef uint2 chunk;
    static chunk load(device const uchar* w, ulong e) { return *(device const uint2*)(w + e); }
    static float scale(device const ushort* s, ulong i) { return bf(s[i]); }
    static float4 lo(chunk c) { return float4(as_type<char4>(c.x)); }
    static float4 hi(chunk c) { return float4(as_type<char4>(c.y)); }
};

// `inv` = 1/rms(X[t]). SWIGLU needs a row pair, so callers apply it themselves.
inline void epilogue(constant MatArgs& a, uint t, uint row, float acc, float inv,
                     device float* Y, device const float* base, device float* aux) {
    ulong o = ulong(t) * a.m + row;
    switch (a.epi) {
    case EPI_NORM: Y[o] = acc * inv * a.scale; break;
    case EPI_RAW: Y[o] = acc; break;
    case EPI_RESID: {
        float v = a.scale * acc;
        Y[o] = a.ra * base[o] + a.rb * v;
        if (a.flags & FLAG_WRITE_RAW) aux[o] = v;
        break;
    }
    case EPI_HEAD: {
        float l = 23.0f * sigmoid((acc + 5.0f) / 7.5f);
        Y[o] = row < a.vocab_real ? l : -INFINITY;
        break;
    }
    }
}

inline void epilogue_swiglu(constant MatArgs& a, uint t, uint row, float g, float u, float inv,
                            device float* Y) {
    g *= inv * a.scale;
    u *= inv * a.scale;
    Y[ulong(t) * (a.m / 2) + row / 2] = g * sigmoid(g) * u;
}

// GEMV: each simdgroup owns ROWS consecutive rows and NB tokens, each lane chunks of 8
// weights of every row. With KS > 1, KS simdgroups share the rows, each taking 1/KS of the
// k chunks, and add up their sums through threadgroup memory. K is a compile-time constant
// so that all of a simdgroup's weight loads are issued before the first is used.
// grid: (ceil(m / (ROWS * 8 / KS)), ceil(n / NB)), 256 threads.
template <typename F, uint ROWS, uint NB, uint K, uint KS>
void gemv_impl(device const uchar* W, device const ushort* S, device const float* X, device float* Y,
               device const float* base, device float* aux, constant MatArgs& a,
               threadgroup float* red, uint2 tg, uint sg, uint lane) {
    constexpr uint CH = K / 256;                  // k chunks per row
    constexpr uint IT = (CH + KS - 1) / KS;       // per simdgroup
    const uint ks = sg % KS, i0 = ks * IT;
    const uint row0 = (tg.x * (8 / KS) + sg / KS) * ROWS;
    if (KS == 1 && row0 >= a.m) return;  // with KS > 1, every simdgroup meets the barrier
    const uint t0 = tg.y * NB;
    typename F::chunk w[ROWS][IT];
    float ws[ROWS][IT];
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; r++) {
        const ulong row = min(row0 + r, a.m - 1);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < IT; i++) {
            if (i0 + i >= CH) break;
            const uint e = (i0 + i) * 256 + lane * 8;
            w[r][i] = F::load(W, row * K + e);
            ws[r][i] = F::scale(S, row * (K >> a.gshift) + (e >> a.gshift));
        }
    }
    float acc[ROWS][NB];
    float ss[NB];
    for (uint b = 0; b < NB; b++) {
        ss[b] = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < ROWS; r++) acc[r][b] = 0.0f;
        device const float* xr = X + ulong(min(t0 + b, a.n - 1)) * K + i0 * 256 + lane * 8;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < IT; i++) {
            if (i0 + i >= CH) break;
            float4 xa = *(device const float4*)(xr + i * 256);
            float4 xb = *(device const float4*)(xr + i * 256 + 4);
            ss[b] += dot(xa, xa) + dot(xb, xb);
            _Pragma("clang loop unroll(full)")
            for (uint r = 0; r < ROWS; r++)
                acc[r][b] += ws[r][i] * (dot(F::lo(w[r][i]), xa) + dot(F::hi(w[r][i]), xb));
        }
    }
    for (uint b = 0; b < NB; b++) {
        ss[b] = simd_sum(ss[b]);
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < ROWS; r++) acc[r][b] = simd_sum(acc[r][b]);
    }
    if (KS > 1) {
        // red: per simdgroup, NB sums of squares then ROWS x NB dot products.
        constexpr uint RS = (ROWS + 1) * NB;
        if (lane == 0) {
            for (uint b = 0; b < NB; b++) {
                red[sg * RS + b] = ss[b];
                _Pragma("clang loop unroll(full)")
                for (uint r = 0; r < ROWS; r++) red[sg * RS + NB + r * NB + b] = acc[r][b];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks != 0 || row0 >= a.m) return;
        for (uint j = 1; j < KS; j++) {
            for (uint b = 0; b < NB; b++) {
                ss[b] += red[(sg + j) * RS + b];
                _Pragma("clang loop unroll(full)")
                for (uint r = 0; r < ROWS; r++) acc[r][b] += red[(sg + j) * RS + NB + r * NB + b];
            }
        }
    }
    if (lane != 0) return;
    for (uint b = 0; b < NB; b++) {
        uint t = t0 + b;
        if (t >= a.n) break;
        float inv = rsqrt(ss[b] / float(K) + RMS_EPS);
        if ((a.flags & FLAG_WRITE_INV) && row0 == 0) aux[t] = inv;
        if (a.epi == EPI_SWIGLU) {
            _Pragma("clang loop unroll(full)")
            for (uint r = 0; r + 1 < ROWS; r += 2)
                if (row0 + r < a.m) epilogue_swiglu(a, t, row0 + r, acc[r][b], acc[r + 1][b], inv, Y);
        } else {
            _Pragma("clang loop unroll(full)")
            for (uint r = 0; r < ROWS; r++)
                if (row0 + r < a.m) epilogue(a, t, row0 + r, acc[r][b], inv, Y, base, aux);
        }
    }
}

#define GEMV(F, FN, R, B, K, KS) \
kernel void gemv_##FN##_r##R##_b##B##_k##K##_s##KS(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], \
                             device float* Y [[buffer(2)]], device const float* base [[buffer(3)]], \
                             device float* aux [[buffer(4)]], constant MatArgs& a [[buffer(5)]], \
                             device const ushort* S [[buffer(8)]], \
                             uint2 tg [[threadgroup_position_in_grid]], \
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float red[KS > 1 ? 8 * (R + 1) * B : 1]; \
    gemv_impl<F, R, B, K, KS>(W, S, X, Y, base, aux, a, red, tg, sg, lane); \
}
// Splitting k: one token only, the engine's case.
#define GEMV_K(F, FN, K) GEMV(F, FN, 1, 1, K, 1) GEMV(F, FN, 1, 2, K, 1) GEMV(F, FN, 1, 4, K, 1) GEMV(F, FN, 1, 8, K, 1) \
                         GEMV(F, FN, 2, 1, K, 1) GEMV(F, FN, 2, 2, K, 1) GEMV(F, FN, 2, 4, K, 1) GEMV(F, FN, 2, 8, K, 1) \
                         GEMV(F, FN, 4, 1, K, 1) GEMV(F, FN, 4, 2, K, 1) GEMV(F, FN, 4, 4, K, 1) GEMV(F, FN, 4, 8, K, 1) \
                         GEMV(F, FN, 1, 1, K, 2) GEMV(F, FN, 2, 1, K, 2) GEMV(F, FN, 4, 1, K, 2) \
                         GEMV(F, FN, 1, 1, K, 4) GEMV(F, FN, 2, 1, K, 4) GEMV(F, FN, 4, 1, K, 4)
GEMV_K(Bf16W, bf16, 1280) GEMV_K(Bf16W, bf16, 3328)
GEMV_K(I8W, i8, 1280) GEMV_K(I8W, i8, 3328)

// GEMM: BM tokens x 64 rows per threadgroup of 4 simdgroups (SM x SN), on fp32 simdgroup
// MMA. k goes in tiles of 16 through <= 10 KB of threadgroup memory, dequantized on the way
// in, with the loads of the next P tiles in flight. Requires m % 64 == 0, k % (16 P) == 0
// and int8 groups of at least 8. grid: (ceil(n / BM), m / 64, ksplit), 128 threads. With
// ksplit > 1, each threadgroup takes one k range of kchunk and writes raw sums, and the
// sums of squares of its X range, for gemm_splitk_reduce.
constant constexpr uint GB = 64;   // rows per threadgroup
constant constexpr uint GK = 16;
constant constexpr uint GS = GK + 4;  // padded row stride, against bank conflicts in the 8x8 loads

// Coordinates of thread_elements() of an 8x8 simdgroup matrix: (row, col) and (row, col + 1).
inline uint2 frag_coord(uint lane) {
    uint qid = lane / 4;
    return uint2((qid & 4) + (lane / 2) % 4, (qid & 2) * 2 + (lane % 2) * 2);
}

template <typename F, uint BM, uint SM, uint SN, uint P>
void gemm_impl(device const uchar* W, device const ushort* S, device const float* X, device float* Y, device const float* base,
               device float* aux, constant MatArgs& a, device float* kpart, device float* kss,
               threadgroup float* Xs, threadgroup float* Wt, threadgroup float* inv_s, uint3 tg,
               uint tid, uint sg, uint lane) {
    constexpr uint WS = GB + 4;          // Wt row stride: the W tile is stored transposed, [k][row]
    constexpr uint TI = BM / SM / 8;     // 8x8 tiles per simdgroup, tokens
    constexpr uint TJ = GB / SN / 8;     // 8x8 tiles per simdgroup, rows
    constexpr uint XL = BM >= 32 ? BM / 32 : 1;  // X float4s per thread per k tile
    constexpr uint TPT = 128 / BM;       // threads per token for the sum of squares
    const uint t0 = tg.x * BM, r0 = tg.y * GB;
    const uint kb = tg.z * a.kchunk, ke = min(kb + a.kchunk, a.k);
    const uint sm = sg / SN, sn = sg % SN;

    simdgroup_float8x8 C[TI][TJ];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < TI; i++)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < TJ; j++) C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Per thread and tile: X float4s (token idx / 4, k (idx % 4) * 4 for idx = tid + 128 i)
    // and one chunk of W (row tid / 2, k (tid % 2) * 8) with its scale.
    const bool xload = tid < BM * 4;
    device const float* xp[XL];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < XL; i++) {
        const uint idx = min(tid + 128 * i, BM * 4 - 1);
        xp[i] = X + ulong(min(t0 + idx / 4, a.n - 1)) * a.k + (idx % 4) * 4;
    }
    const uint wr = tid >> 1, wk = (tid & 1) * 8;
    const ulong we = ulong(r0 + wr) * a.k + wk;          // element of W at k = 0
    const ulong se = ulong(r0 + wr) * (a.k >> a.gshift);  // and of its scales
    float4 rx[P][XL];
    typename F::chunk rw[P];
    float rs[P];
    _Pragma("clang loop unroll(full)")
    for (uint p = 0; p < P; p++) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < XL; i++) rx[p][i] = *(device const float4*)(xp[i] + kb + p * GK);
        rw[p] = F::load(W, we + kb + p * GK);
        rs[p] = F::scale(S, se + ((kb + p * GK + wk) >> a.gshift));
    }
    float ss = 0.0f;  // squares of token tid / TPT, part tid % TPT of each tile

    for (uint k1 = kb; k1 < ke; k1 += P * GK) {
        _Pragma("clang loop unroll(full)")
        for (uint p = 0; p < P; p++) {
            const uint k0 = k1 + p * GK;
            if (xload) {
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < XL; i++) {
                    const uint idx = tid + 128 * i;
                    *(threadgroup float4*)(Xs + (idx / 4) * GS + (idx % 4) * 4) = rx[p][i];
                }
            }
            {
                float4 lo = F::lo(rw[p]) * rs[p], hi = F::hi(rw[p]) * rs[p];
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < 4; i++) {
                    Wt[(wk + i) * WS + wr] = lo[i];
                    Wt[(wk + 4 + i) * WS + wr] = hi[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k0 + P * GK < ke) {
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < XL; i++) rx[p][i] = *(device const float4*)(xp[i] + k0 + P * GK);
                rw[p] = F::load(W, we + k0 + P * GK);
                rs[p] = F::scale(S, se + ((k0 + P * GK + wk) >> a.gshift));
            }
            {
                threadgroup const float* xs = Xs + (tid / TPT) * GS + (tid % TPT) * (GK / TPT);
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < GK / TPT; i++) ss += xs[i] * xs[i];
            }
            _Pragma("clang loop unroll(full)")
            for (uint kk = 0; kk < GK; kk += 8) {
                simdgroup_float8x8 A[TI], B[TJ];
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < TI; i++) simdgroup_load(A[i], Xs + (sm * TI * 8 + i * 8) * GS + kk, GS);
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < TJ; j++) simdgroup_load(B[j], Wt + kk * WS + sn * TJ * 8 + j * 8, WS);
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < TI; i++)
                    _Pragma("clang loop unroll(full)")
                    for (uint j = 0; j < TJ; j++) simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    _Pragma("clang loop unroll(full)")
    for (ushort o = 1; o < TPT; o <<= 1) ss += simd_shuffle_xor(ss, o);
    const uint2 fc = frag_coord(lane);
    if (a.ksplit > 1) {
        if (tg.y == 0 && tid % TPT == 0 && t0 + tid / TPT < a.n) kss[ulong(tg.z) * a.n + t0 + tid / TPT] = ss;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < TI; i++) {
            const uint tt = sm * TI * 8 + i * 8 + fc.x;
            if (t0 + tt >= a.n) continue;
            device float* kp = kpart + (ulong(tg.z) * a.n + t0 + tt) * a.m + r0 + sn * TJ * 8 + fc.y;
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < TJ; j++)
                *(device float2*)(kp + j * 8) = float2(C[i][j].thread_elements()[0], C[i][j].thread_elements()[1]);
        }
        return;
    }
    if (tid % TPT == 0) inv_s[tid / TPT] = rsqrt(ss / float(a.k) + RMS_EPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((a.flags & FLAG_WRITE_INV) && r0 == 0 && tid < BM && t0 + tid < a.n) aux[t0 + tid] = inv_s[tid];

    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < TI; i++) {
        const uint tt = sm * TI * 8 + i * 8 + fc.x;
        if (t0 + tt >= a.n) continue;
        const float inv = inv_s[tt];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < TJ; j++) {
            const uint rr = r0 + sn * TJ * 8 + j * 8 + fc.y;
            const float c0 = C[i][j].thread_elements()[0], c1 = C[i][j].thread_elements()[1];
            if (a.epi == EPI_SWIGLU) {
                epilogue_swiglu(a, t0 + tt, rr, c0, c1, inv, Y);  // rr even: a (gate, up) pair
            } else {
                epilogue(a, t0 + tt, rr, c0, inv, Y, base, aux);
                epilogue(a, t0 + tt, rr + 1, c1, inv, Y, base, aux);
            }
        }
    }
}

#define GEMM(F, FN, BM, SM, SN, P) \
kernel void gemm_##FN##_m##BM##_p##P(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], \
                       device float* Y [[buffer(2)]], device const float* base [[buffer(3)]], \
                       device float* aux [[buffer(4)]], constant MatArgs& a [[buffer(5)]], \
                       device float* kpart [[buffer(6)]], device float* kss [[buffer(7)]], \
                       device const ushort* S [[buffer(8)]], \
                       uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float Xs[BM * GS]; \
    threadgroup float Wt[GK * (GB + 4)]; \
    threadgroup float inv_s[BM]; \
    gemm_impl<F, BM, SM, SN, P>(W, S, X, Y, base, aux, a, kpart, kss, Xs, Wt, inv_s, tg, tid, sg, lane); \
}
#define GEMM_F(F, FN) GEMM(F, FN, 64, 2, 2, 1) GEMM(F, FN, 64, 2, 2, 2) GEMM(F, FN, 64, 2, 2, 4) \
                      GEMM(F, FN, 32, 1, 4, 1) GEMM(F, FN, 32, 1, 4, 2) GEMM(F, FN, 32, 1, 4, 4) \
                      GEMM(F, FN, 16, 1, 4, 1) GEMM(F, FN, 16, 1, 4, 2) GEMM(F, FN, 16, 1, 4, 4)
GEMM_F(Bf16W, bf16)
GEMM_F(I8W, i8)

// Sums the split-K partials in a fixed order, then applies the epilogue. One thread per
// (row pair, token): grid (ceil(m / 512), n), 256 threads.
constant constexpr uint MAX_KSPLIT = 8;
kernel void gemm_splitk_reduce(device const float* kpart [[buffer(0)]], device const float* kss [[buffer(1)]],
                               device float* Y [[buffer(2)]], device const float* base [[buffer(3)]],
                               device float* aux [[buffer(4)]], constant MatArgs& a [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint row = gid.x * 2, t = gid.y;
    if (row >= a.m || t >= a.n) return;
    float s8[MAX_KSPLIT];
    float2 c8[MAX_KSPLIT];
    _Pragma("clang loop unroll(full)")
    for (uint z = 0; z < MAX_KSPLIT; z++) {
        const uint zz = min(z, a.ksplit - 1);
        s8[z] = kss[zz * a.n + t];
        c8[z] = *(device const float2*)(kpart + (ulong(zz) * a.n + t) * a.m + row);
    }
    float s = 0.0f;
    float2 c = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint z = 0; z < MAX_KSPLIT; z++) {
        if (z < a.ksplit) { s += s8[z]; c += c8[z]; }
    }
    const float inv = rsqrt(s / float(a.k) + RMS_EPS);
    if ((a.flags & FLAG_WRITE_INV) && row == 0) aux[t] = inv;
    if (a.epi == EPI_SWIGLU) {
        epilogue_swiglu(a, t, row, c.x, c.y, inv, Y);
    } else {
        epilogue(a, t, row, c.x, inv, Y, base, aux);
        epilogue(a, t, row + 1, c.y, inv, Y, base, aux);
    }
}

// Streaming GEMM, for 2-16 tokens: the GEMV's weight streaming, with the products on the
// matrix units. C^T (rows x tokens) = W (rows x k) . X^T (k x tokens), in 8x8 fragments
// built in registers, with no threadgroup staging and no barrier in the k loop.
// - Within each 32-k block, fragment column i of block v is physical k 4 i + v, so each
//   lane loads 8 consecutive weights of one row and 4 consecutive inputs of two tokens.
// - Each simdgroup owns R blocks of 8 rows, NT tiles of 8 tokens and 1/KS of the k blocks.
//   The KS simdgroups of a row group add up their sums through threadgroup memory.
// grid: (m / (64 R / KS), ceil(n / (8 NT))), 256 threads.
template <typename F, uint R, uint NT, uint K, uint KS>
void gemm_stream_impl(device const uchar* W, device const ushort* S, device const float* X, device float* Y,
                      device const float* base, device float* aux, constant MatArgs& a,
                      threadgroup float* red, uint2 tg, uint sg, uint lane) {
    constexpr uint SB = K / 32;               // k blocks per row
    constexpr uint IT = (SB + KS - 1) / KS;   // per simdgroup
    const uint ks = sg % KS, b0 = ks * IT;
    const uint row0 = (tg.x * (8 / KS) + sg / KS) * R * 8;
    const uint t0 = tg.y * NT * 8;
    const uint2 fc = frag_coord(lane);  // (r, c)

    typename F::chunk w[R][IT];
    float ws[R][IT];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < R; j++) {
        const ulong row = min(row0 + j * 8 + fc.x, a.m - 1);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < IT; i++) {
            if (b0 + i >= SB) break;
            const uint e = (b0 + i) * 32 + fc.y * 4;
            w[j][i] = F::load(W, row * K + e);
            ws[j][i] = F::scale(S, row * (K >> a.gshift) + (e >> a.gshift));
        }
    }
    device const float* xp[NT][2];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < NT; u++)
        for (uint b = 0; b < 2; b++) xp[u][b] = X + ulong(min(t0 + u * 8 + fc.y + b, a.n - 1)) * K + fc.x * 4;

    simdgroup_float8x8 C[R][NT];
    float ss[NT][2];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < NT; u++) {
        ss[u][0] = ss[u][1] = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < R; j++) C[j][u] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float4 xn[NT][2];  // the next k block's inputs
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < NT; u++)
        for (uint b = 0; b < 2; b++) xn[u][b] = *(device const float4*)(xp[u][b] + b0 * 32);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < IT; i++) {
        if (b0 + i >= SB) break;
        float4 xv[NT][2];
        _Pragma("clang loop unroll(full)")
        for (uint u = 0; u < NT; u++)
            _Pragma("clang loop unroll(full)")
            for (uint b = 0; b < 2; b++) {
                xv[u][b] = xn[u][b];
                if (i + 1 < IT && b0 + i + 1 < SB) xn[u][b] = *(device const float4*)(xp[u][b] + (b0 + i + 1) * 32);
                ss[u][b] += dot(xv[u][b], xv[u][b]);
            }
        float4 lo[R], hi[R];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < R; j++) {
            lo[j] = F::lo(w[j][i]) * ws[j][i];
            hi[j] = F::hi(w[j][i]) * ws[j][i];
        }
        _Pragma("clang loop unroll(full)")
        for (uint v = 0; v < 4; v++) {
            simdgroup_float8x8 A[R], B[NT];
            _Pragma("clang loop unroll(full)")
            for (uint u = 0; u < NT; u++) {
                B[u].thread_elements()[0] = xv[u][0][v];
                B[u].thread_elements()[1] = xv[u][1][v];
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < R; j++) {
                A[j].thread_elements()[0] = lo[j][v];
                A[j].thread_elements()[1] = hi[j][v];
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < R; j++)
                _Pragma("clang loop unroll(full)")
                for (uint u = 0; u < NT; u++) simdgroup_multiply_accumulate(C[j][u], A[j], B[u], C[j][u]);
        }
    }
    // c[j][u][b]: row row0 + 8 j + r, token t0 + 8 u + c + b.
    float c[R][NT][2];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < R; j++)
        _Pragma("clang loop unroll(full)")
        for (uint u = 0; u < NT; u++) c[j][u][0] = C[j][u].thread_elements()[0], c[j][u][1] = C[j][u].thread_elements()[1];
    // Sums of squares: over the lanes of the other fragment rows (lane bits 1, 2 and 4).
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < NT; u++)
        for (uint b = 0; b < 2; b++) {
            ss[u][b] += simd_shuffle_xor(ss[u][b], 2);
            ss[u][b] += simd_shuffle_xor(ss[u][b], 4);
            ss[u][b] += simd_shuffle_xor(ss[u][b], 16);
        }
    if (KS > 1) {
        // Per simdgroup: R x NT x 64 partial sums, then NT x 8 sums of squares.
        constexpr uint RS = (R * 64 + 8) * NT;
        if (ks != 0) {
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < R; j++)
                _Pragma("clang loop unroll(full)")
                for (uint u = 0; u < NT; u++)
                    *(threadgroup float2*)(red + sg * RS + (j * NT + u) * 64 + lane * 2) = float2(c[j][u][0], c[j][u][1]);
            if (fc.x == 0)
                _Pragma("clang loop unroll(full)")
                for (uint u = 0; u < NT; u++)
                    *(threadgroup float2*)(red + sg * RS + R * NT * 64 + u * 8 + fc.y) = float2(ss[u][0], ss[u][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks != 0) return;
        for (uint q = 1; q < KS; q++) {
            _Pragma("clang loop unroll(full)")
            for (uint u = 0; u < NT; u++) {
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < R; j++) {
                    const float2 p = *(threadgroup const float2*)(red + (sg + q) * RS + (j * NT + u) * 64 + lane * 2);
                    c[j][u][0] += p.x;
                    c[j][u][1] += p.y;
                }
                const float2 p = *(threadgroup const float2*)(red + (sg + q) * RS + R * NT * 64 + u * 8 + fc.y);
                ss[u][0] += p.x;
                ss[u][1] += p.y;
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < NT; u++) {
        float inv[2];
        for (uint b = 0; b < 2; b++) {
            inv[b] = rsqrt(ss[u][b] / float(K) + RMS_EPS);
            const uint t = t0 + u * 8 + fc.y + b;
            if ((a.flags & FLAG_WRITE_INV) && row0 == 0 && fc.x == 0 && t < a.n) aux[t] = inv[b];
        }
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < R; j++) {
            const uint row = row0 + j * 8 + fc.x;
            if (a.epi == EPI_SWIGLU) {
                // (gate, up) are rows (2 i, 2 i + 1): fragment rows r and r ^ 1, lane bit 1.
                const float up[2] = {simd_shuffle_xor(c[j][u][0], 2), simd_shuffle_xor(c[j][u][1], 2)};
                if (fc.x & 1 || row >= a.m) continue;
                for (uint b = 0; b < 2; b++) {
                    const uint t = t0 + u * 8 + fc.y + b;
                    if (t < a.n) epilogue_swiglu(a, t, row, c[j][u][b], up[b], inv[b], Y);
                }
            } else if (row < a.m) {
                for (uint b = 0; b < 2; b++) {
                    const uint t = t0 + u * 8 + fc.y + b;
                    if (t < a.n) epilogue(a, t, row, c[j][u][b], inv[b], Y, base, aux);
                }
            }
        }
    }
}

#define GEMM_STREAM(F, FN, R, NT, K, KS) \
kernel void gemms_##FN##_r##R##_t##NT##_k##K##_s##KS(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], \
                             device float* Y [[buffer(2)]], device const float* base [[buffer(3)]], \
                             device float* aux [[buffer(4)]], constant MatArgs& a [[buffer(5)]], \
                             device const ushort* S [[buffer(8)]], \
                             uint2 tg [[threadgroup_position_in_grid]], \
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float red[KS > 1 ? 8 * (R * 64 + 8) * NT : 1]; \
    gemm_stream_impl<F, R, NT, K, KS>(W, S, X, Y, base, aux, a, red, tg, sg, lane); \
}
#define GEMM_STREAM_K(F, FN, K) \
    GEMM_STREAM(F, FN, 1, 1, K, 4) GEMM_STREAM(F, FN, 1, 1, K, 8) GEMM_STREAM(F, FN, 2, 1, K, 4) GEMM_STREAM(F, FN, 2, 1, K, 8) \
    GEMM_STREAM(F, FN, 1, 2, K, 4) GEMM_STREAM(F, FN, 1, 2, K, 8) GEMM_STREAM(F, FN, 2, 2, K, 4) GEMM_STREAM(F, FN, 2, 2, K, 8)
GEMM_STREAM_K(Bf16W, bf16, 1280) GEMM_STREAM_K(Bf16W, bf16, 3328)
GEMM_STREAM_K(I8W, i8, 1280) GEMM_STREAM_K(I8W, i8, 3328)

// --------------------------------------------------------------- KV layout
// Local layers: a ring per (layer, sequence slot): [nlocal][max_seqs][NKV][ring][D].
// Global layers: pages shared by all sequences: [nglobal][n_pages][NKV][page][D],
// with a page table per sequence slot: page_table[seq * max_pages + pos / page].
struct KvArgs {
    uint is_global, kv_layer;
    uint ring, max_seqs;
    uint page_shift, n_pages, max_pages;
    uint window;  // keys visible to a local query besides itself (1024)
};

// Where the K (or V) of kv head g at cache position pos starts, in KV layer `layer` of kv's kind.
inline ulong kv_row_of(constant KvArgs& kv, uint layer, device const uint* page_table, uint seq, uint g, uint pos) {
    if (kv.is_global) {
        uint page = page_table[seq * kv.max_pages + (pos >> kv.page_shift)];
        uint psize = 1u << kv.page_shift;
        return ((ulong(layer) * kv.n_pages + page) * NKV + g) * psize * D + ulong(pos & (psize - 1)) * D;
    }
    return ((ulong(layer) * kv.max_seqs + seq) * NKV + g) * kv.ring * D + ulong(pos % kv.ring) * D;
}

inline ulong kv_row(constant KvArgs& kv, device const uint* page_table, uint seq, uint g, uint pos) {
    return kv_row_of(kv, kv.kv_layer, page_table, seq, g, pos);
}

// ----------------------------------------------------- after the qkv matmul
// Per token: q/k rms-norm and RoPE, value embeddings, K/V into the cache, the XSA value
// direction and the attention gates. One threadgroup per token, 12 simdgroups:
// 0-9 query heads, 10-11 kv heads. Each lane owns 4 of the 128 dims.
struct PostArgs {
    uint rope, ve;
};

// rms_norm over the head, then RoPE on dims 0..63 as 32 interleaved pairs. `cs` holds this
// token's (cos, sin) per pair, rounded to bf16 as HF does.
inline float4 norm_rope(float4 v, bool rope, uint lane, device const float2* cs) {
    float ss = simd_sum(dot(v, v));
    v *= rsqrt(ss / float(D) + RMS_EPS);
    if (rope && lane < 16) {
        float4 c = *(device const float4*)(cs + lane * 2);  // (cos0, sin0, cos1, sin1)
        float a0 = v.x, a1 = v.y, b0 = v.z, b1 = v.w;
        v = float4(c.x * a0 + c.y * a1, c.x * a1 - c.y * a0, c.z * b0 + c.w * b1, c.z * b1 - c.w * b0);
    }
    return v;
}

// v + 2 sigmoid(rms_norm(x)[:12] . ve_gate[g]) * value_embed[token][g].
inline float4 add_ve(float4 v, device const float* xt, float inv, uint token, uint g, uint lane,
                     device const ushort* ve_table, device const float* ve_gate) {
    const uint d0 = lane * 4;
    float s = 0.0f;
    if (lane < 3) s = dot(*(device const float4*)(xt + d0) * inv, *(device const float4*)(ve_gate + g * 12 + d0));
    float gate = 2.0f * sigmoid(simd_sum(s));
    device const ushort* ve = ve_table + ulong(token) * VE_W + g * D + d0;
    return v + gate * float4(bf(ve[0]), bf(ve[1]), bf(ve[2]), bf(ve[3]));
}

kernel void qkv_post(device float* qkv [[buffer(0)]],
                     device const float* x [[buffer(1)]],          // attention input before the norm
                     device const float* inv_rms [[buffer(2)]],
                     device const uint2* meta [[buffer(3)]],
                     device const uint* tokens [[buffer(4)]],
                     device const ushort* ve_table [[buffer(5)]],
                     device const float* ve_gate [[buffer(6)]],    // [2][12]
                     device const float* attn_gate [[buffer(7)]],  // [10][128]
                     device ushort* k_cache [[buffer(8)]],
                     device ushort* v_cache [[buffer(9)]],
                     device float* vdir [[buffer(10)]],            // [n][2][128]
                     device float* gates [[buffer(11)]],           // [n][10]
                     device const uint* page_table [[buffer(12)]],
                     device const float2* rope_tab [[buffer(13)]],  // [position][32] (cos, sin)
                     constant KvArgs& kv [[buffer(14)]],
                     constant PostArgs& pa [[buffer(15)]],
                     device const uint2* tin [[buffer(16)]],
                     uint t [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const uint seq = meta[t].x, pos = meta[t].y;
    const float inv = inv_rms[t];
    device const float* xt = x + ulong(t) * H;
    device const float2* cs = rope_tab + ulong(pos) * 32;
    const uint d0 = lane * 4;

    if (sg < NQ) {
        device float* q = qkv + ulong(t) * QKV + sg * D + d0;
        *(device float4*)q = norm_rope(*(device float4*)q, pa.rope, lane, cs);
        // Attention gate: 2 sigmoid(rms_norm(x)[:128] . attn_gate[h]).
        float4 xg = *(device const float4*)(xt + d0) * inv;
        float s = simd_sum(dot(xg, *(device const float4*)(attn_gate + sg * D + d0)));
        if (lane == 0) gates[t * NQ + sg] = 2.0f * sigmoid(s);
        return;
    }
    const uint g = sg - NQ;
    device float* k = qkv + ulong(t) * QKV + KOFF + g * D + d0;
    device float* v = qkv + ulong(t) * QKV + VOFF + g * D + d0;
    const ulong row = kv_row(kv, page_table, seq, g, cache_pos(meta, tin, t)) + d0;

    float4 kk = norm_rope(*(device float4*)k, pa.rope, lane, cs);
    *(device float4*)k = kk;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) k_cache[row + i] = to_bf(kk[i]);

    float4 vv = *(device float4*)v;
    if (pa.ve) vv = add_ve(vv, xt, inv, tokens[t], g, lane, ve_table, ve_gate);
    *(device float4*)v = vv;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) v_cache[row + i] = to_bf(vv[i]);
    float nrm = sqrt(simd_sum(dot(vv, vv)));
    *(device float4*)(vdir + (ulong(t) * NKV + g) * D + d0) = vv / max(nrm, XSA_EPS);
}

// ---------------------------------------------------------------- attention
// q and k are rms-normed without gain, so the scaled scores stay within +-12.8: the softmax
// needs no running max, and splits merge by plain sums. Partials are (sum p, sum p v) per
// (token, head, split).

// The first key the query at `pos` sees.
inline uint key_lo(constant KvArgs& kv, uint pos) {
    return kv.is_global ? 0u : (pos > kv.window ? pos - kv.window : 0u);
}

struct AttnArgs {
    uint split;        // cached keys per split, a multiple of 32
    uint max_splits;
    float scale_log2;  // softmax scale * log2(e)
    uint rope, ve, capture;
    uint fixed_splits; // attn_sum_reduce: splits to sum per token (attn_mma's max_splits)
    uint col_sets;     // attn_verify: column groups of the longest run
    uint one_round;    // attn_verify: its simdgroups add up their sums in one round
};

// Both attention kernels, per simdgroup and batch of 8 keys, for 8 query columns
// (attn_verify: (token, head) pairs; attn_mma: tokens of one head):
//   S^T (keys x cols) = K (keys x dims) . Q^T (dims x cols)   16 MMAs over the dims
//   P^T = exp2(S^T), in place: an accumulator has the layout of a B operand
//   O^T (dims x cols) += V^T (dims x keys) . P^T              16 MMAs, 8 dims each
// The dims are permuted within each fragment so that every lane loads 64 contiguous bytes
// of K and of V.
inline float bf_lo(uint w) { return as_type<float>(w << 16); }
inline float bf_hi(uint w) { return as_type<float>(w & 0xffff0000u); }

// S^T's 16 MMAs over the dims go into this many independent accumulators, so that the
// matrix units don't wait on one chain.
constant constexpr uint QK_CHAINS = 4;
inline float2 qk_sum(thread const simdgroup_float8x8* S) {
    float2 s = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < QK_CHAINS; j++) s += float2(S[j].thread_elements()[0], S[j].thread_elements()[1]);
    return s;
}

// This lane's K and V words for the batch of 8 keys from kb (clamped to the last valid key).
// Global batches start at multiples of 8, so inside one page; local ones may wrap the ring.
inline void load_batch_mma(constant KvArgs& kv, device const uint* page_table, device const ushort* k_cache,
                           device const ushort* v_cache, uint seq, uint g, uint kb, uint k_end, uint fr,
                           uint cp, thread uint4* kw, thread uint4* vw) {
    const uint cnt = min(8u, k_end - kb);
    const ulong r0 = kv_row(kv, page_table, seq, g, kb);
    const uint wrap_at = kv.is_global ? 8u : kv.ring - kb % kv.ring;
    const ulong ring_rows = ulong(kv.ring) * D;
    const uint ik = min(fr, cnt - 1);
    device const uint4* kp = (device const uint4*)(k_cache + r0 + ulong(ik) * D - (ik >= wrap_at ? ring_rows : 0) + cp * 32);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) kw[i] = kp[i];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 2; j++) {
        const uint iv = min(2 * cp + j, cnt - 1);
        device const uint4* vp = (device const uint4*)(v_cache + r0 + ulong(iv) * D - (iv >= wrap_at ? ring_rows : 0) + fr * 16);
        vw[2 * j] = vp[0];
        vw[2 * j + 1] = vp[1];
    }
}

// load_batch_mma's words with the dims permuted so that each load instruction's lanes read
// fewer cache lines: kw[i] is dims 32 i + 8 cp..+7 of key fr, vw[2 j + u] dims
// 64 u + 8 fr..+7 of key 2 cp + j. dim_qk gives the dim of the A operands' k-slot j of
// fragment f, and dim_o that of O^T fragment i's row r.
inline uint dim_qk(uint f, uint j) { return 32 * (f / 4) + 8 * (j / 2) + 2 * (f % 4) + j % 2; }
inline uint dim_o(uint i, uint r) { return 64 * (i / 8) + 8 * r + i % 8; }
inline void load_batch_lines(constant KvArgs& kv, device const uint* page_table, device const ushort* k_cache,
                             device const ushort* v_cache, uint seq, uint g, uint kb, uint k_end, uint fr,
                             uint cp, thread uint4* kw, thread uint4* vw) {
    const uint cnt = min(8u, k_end - kb);
    const ulong r0 = kv_row(kv, page_table, seq, g, kb);
    const uint wrap_at = kv.is_global ? 8u : kv.ring - kb % kv.ring;
    const ulong ring_rows = ulong(kv.ring) * D;
    const uint ik = min(fr, cnt - 1);
    device const uint4* kp = (device const uint4*)(k_cache + r0 + ulong(ik) * D - (ik >= wrap_at ? ring_rows : 0)) + cp;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) kw[i] = kp[4 * i];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 2; j++) {
        const uint iv = min(2 * cp + j, cnt - 1);
        device const uint4* vp = (device const uint4*)(v_cache + r0 + ulong(iv) * D - (iv >= wrap_at ? ring_rows : 0)) + fr;
        vw[2 * j] = vp[0];
        vw[2 * j + 1] = vp[8];
    }
}

// ------------------------------------------------------ small-run attention
// For steps whose runs all have at most 8 tokens (decode, verification, the drafter's
// blocks), attn_verify replaces qkv_post and attn_mma, and reads each sequence's cached KV
// once for all of its run's tokens. One threadgroup per (run, kv head, key split):
//   - The run's T tokens x 5 query heads are packed into the MMA columns: column
//     p = 5 i + h is token i, head h, in C = ceil(5 T / 8) groups of 8.
//   - Simdgroup sg takes column group c = sg % C and the key batches b = sg / C, + KI, ...;
//     the KI simdgroups of a group add up their sums through threadgroup memory, in one
//     round (`one_round`) or halving each round.
//   - The cached keys [key_lo(p0), p0) are split across threadgroups, p0 being the run's
//     first position. The run's T new keys get the last split, whose threadgroup first
//     processes them (norm, RoPE, value embeddings, cache writes at p0 + row, XSA
//     directions, gates), then attends to them under each token's visibility mask `vis`.
//     Each token's window is its own position's: a draft tree's rows needn't be consecutive.
// Threadgroup memory is sized per dispatch: tgm holds Q^T (16 x 64 floats per column group)
// and the new keys and values (bf16, 8 x 128 each), then the simdgroups' sums; tls their l.
constant constexpr uint VERIFY_T = 8;  // tokens per run, at most

// A new key and value of the run: token t's kv head g at position pos.
inline void new_key(device float* qkv_t, device const float* x, device const float* inv_rms, device const uint* tokens,
                    device const ushort* ve_table, device const float* ve_gate, device const float* attn_gate,
                    device ushort* k_cache, device ushort* v_cache, device const uint* page_table,
                    device const float2* cs, device float* vdir, device float* gates, constant KvArgs& kv,
                    constant AttnArgs& args, uint t, uint g, uint seq, uint pos, uint lane,
                    threadgroup ushort* nk, threadgroup ushort* nv) {
    const uint d0 = lane * 4;
    device const float* xt = x + ulong(t) * H;
    const float inv = inv_rms[t];
    device float* kp = qkv_t + KOFF + g * D + d0;
    device float* vp = qkv_t + VOFF + g * D + d0;
    const float4 kc = norm_rope(*(device const float4*)kp, args.rope, lane, cs);
    float4 vc = *(device const float4*)vp;
    if (args.ve) vc = add_ve(vc, xt, inv, tokens[t], g, lane, ve_table, ve_gate);
    const ulong row = kv_row(kv, page_table, seq, g, pos) + d0;
    ushort4 kb16, vb16;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) { kb16[i] = to_bf(kc[i]); vb16[i] = to_bf(vc[i]); }
    *(device ushort4*)(k_cache + row) = kb16;
    *(device ushort4*)(v_cache + row) = vb16;
    *(threadgroup ushort4*)(nk + d0) = kb16;
    *(threadgroup ushort4*)(nv + d0) = vb16;
    if (args.capture) { *(device float4*)kp = kc; *(device float4*)vp = vc; }
    const float nrm = sqrt(simd_sum(dot(vc, vc)));
    *(device float4*)(vdir + (ulong(t) * NKV + g) * D + d0) = vc / max(nrm, XSA_EPS);
    const float4 xg = *(device const float4*)(xt + d0) * inv;
    _Pragma("clang loop unroll(full)")
    for (uint h = 0; h < GROUP; h++) {
        const uint hh = g * GROUP + h;
        const float gs = simd_sum(dot(xg, *(device const float4*)(attn_gate + hh * D + d0)));
        if (lane == 0) gates[t * NQ + hh] = 2.0f * sigmoid(gs);
    }
}

// load_batch_mma's words for the run's new keys, from their bf16 copies in threadgroup
// memory (nk, nv: [key][128]).
inline void load_new_mma(threadgroup const ushort* nk, threadgroup const ushort* nv, uint cnt, uint fr, uint cp,
                         thread uint4* kw, thread uint4* vw) {
    const uint ik = min(fr, cnt - 1);
    threadgroup const uint4* kp = (threadgroup const uint4*)(nk + ik * D + cp * 32);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) kw[i] = kp[i];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 2; j++) {
        threadgroup const uint4* vp = (threadgroup const uint4*)(nv + min(2 * cp + j, cnt - 1) * D + fr * 16);
        vw[2 * j] = vp[0];
        vw[2 * j + 1] = vp[1];
    }
}

// attn_verify's batches of 8 keys from k_begin, + 8 STRIDE, ... below k_end, for one
// column group (Q^T at qc). This lane's two columns see keys from lo[e], or with NEW, the
// run's new keys in vis[e]; padding columns (ok false) see nothing.
template <bool NEW, uint STRIDE>
inline void verify_batches(constant KvArgs& kv, device const uint* page_table, device const ushort* k_cache,
                           device const ushort* v_cache, uint seq, uint g, uint k_begin, uint k_end, uint b0, uint fr,
                           uint cp, threadgroup const float* qc, threadgroup const ushort* nk, threadgroup const ushort* nv,
                           uint2 lo, uint2 vis, bool2 ok, thread simdgroup_float8x8* O, thread float2& lsum) {
    const uint nb = (k_end - k_begin + 7) / 8;
    for (uint b = b0; b < nb; b += STRIDE) {
        const uint kb = k_begin + b * 8;
        uint4 kw[4], vw[4];
        if (NEW) load_new_mma(nk, nv, k_end - kb, fr, cp, kw, vw);
        else load_batch_mma(kv, page_table, k_cache, v_cache, seq, g, kb, k_end, fr, cp, kw, vw);
        const uint kp = kb + fr;  // this lane's key
        const bool kin = kp < k_end;
        simdgroup_float8x8 S[QK_CHAINS];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < QK_CHAINS; j++) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)")
        for (uint f = 0; f < 16; f++) {
            simdgroup_float8x8 A, B;
            const uint w = kw[f / 4][f % 4];
            A.thread_elements()[0] = bf_lo(w);
            A.thread_elements()[1] = bf_hi(w);
            simdgroup_load(B, qc + f * 64, 8);
            simdgroup_multiply_accumulate(S[f % QK_CHAINS], A, B, S[f % QK_CHAINS]);
        }
        const float2 sc = qk_sum(S);
        const bool va = kin && (NEW ? bool((vis.x >> fr) & 1u) : ok.x && kp >= lo.x);
        const bool vb = kin && (NEW ? bool((vis.y >> fr) & 1u) : ok.y && kp >= lo.y);
        const float pa = va ? exp2(sc.x) : 0.0f;
        const float pb = vb ? exp2(sc.y) : 0.0f;
        lsum += float2(pa, pb);
        simdgroup_float8x8 P;
        P.thread_elements()[0] = pa;
        P.thread_elements()[1] = pb;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16; i++) {
            simdgroup_float8x8 A;
            const uint wa = vw[i / 8][(i % 8) / 2], wb = vw[2 + i / 8][(i % 8) / 2];
            A.thread_elements()[0] = i % 2 ? bf_hi(wa) : bf_lo(wa);
            A.thread_elements()[1] = i % 2 ? bf_hi(wb) : bf_lo(wb);
            simdgroup_multiply_accumulate(O[i], A, P, O[i]);
        }
    }
}

template <uint KI>
void attn_verify_impl(device float* qkv, device const float* x, device const float* inv_rms, device const uint2* meta,
                      device const uint* tokens, device const ushort* ve_table, device const float* ve_gate,
                      device const float* attn_gate, device ushort* k_cache, device ushort* v_cache,
                      device const uint* page_table, device const float2* rope_tab, device float* part,
                      device float* vdir, device float* gates, constant KvArgs& kv, constant AttnArgs& args,
                      device const uint2* runs, device const uint* vis, device float* cap_q, uint3 gid, uint sg,
                      uint nsg, uint lane, threadgroup float* tgm, threadgroup float* tls) {
    const uint s = gid.x, g = gid.y;
    const uint t0 = runs[gid.z].x, T = runs[gid.z].y;
    const uint seq = meta[t0].x, p0 = meta[t0].y;
    const uint lo0 = key_lo(kv, p0);
    const uint nsplit = (p0 - lo0 + args.split - 1) / args.split + 1;
    if (s >= nsplit) return;
    const bool newest = s == nsplit - 1;
    const uint C = args.col_sets;  // column groups of the dispatch's longest run
    const uint c = sg % C, ki = sg / C;
    const uint d0 = lane * 4;

    // Q^T (B operand) of every column group, normed and roped: k-slot r of fragment f is dim
    // 32 (r / 2) + 2 f + r % 2, at tgm[(c * 16 + f) * 64 + r * 8 + column].
    for (uint p = sg; p < C * 8; p += nsg) {
        const uint i = p / GROUP, hq = g * GROUP + p % GROUP;
        float4 q = 0.0f;
        if (i < T) {
            device const float* qp = qkv + ulong(t0 + i) * QKV + hq * D + d0;
            q = norm_rope(*(device const float4*)qp, args.rope, lane, rope_tab + ulong(meta[t0 + i].y) * 32);
            if (args.capture && s == 0) *(device float4*)(cap_q + ulong(t0 + i) * H + hq * D + d0) = q;
            q *= args.scale_log2;
        }
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 4; e++) {
            const uint d = d0 + e;
            tgm[((p / 8) * 16 + (d % 32) / 2) * 64 + (2 * (d / 32) + d % 2) * 8 + p % 8] = q[e];
        }
    }
    // The new keys' bf16 copies, after Q^T.
    threadgroup ushort* nk = (threadgroup ushort*)(tgm + C * 16 * 64);
    threadgroup ushort* nv = nk + VERIFY_T * D;
    if (newest) {
        for (uint i = sg; i < T; i += nsg)
            new_key(qkv + ulong(t0 + i) * QKV, x, inv_rms, tokens, ve_table, ve_gate, attn_gate, k_cache, v_cache,
                    page_table, rope_tab + ulong(meta[t0 + i].y) * 32, vdir, gates, kv, args, t0 + i, g, seq, p0 + i,
                    lane, nk + i * D, nv + i * D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint2 fc = frag_coord(lane);
    const uint fr = fc.x, fcol = fc.y, cp = fcol / 2;
    // This lane's two columns: their token's first visible cached key, and its mask over the
    // new keys.
    const uint ia = (c * 8 + fcol) / GROUP, ib = (c * 8 + fcol + 1) / GROUP;
    const bool2 ok = bool2(ia < T, ib < T);
    const uint2 lo = uint2(key_lo(kv, meta[t0 + min(ia, T - 1)].y), key_lo(kv, meta[t0 + min(ib, T - 1)].y));
    const uint2 vmask = uint2(ok.x ? vis[t0 + ia] : 0u, ok.y ? vis[t0 + ib] : 0u);
    simdgroup_float8x8 O[16];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 16; i++) O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float2 lsum = 0.0f;
    threadgroup const float* qc = tgm + c * 16 * 64;
    if (newest) {
        if (ki == 0) verify_batches<true, 1>(kv, page_table, k_cache, v_cache, seq, g, p0, p0 + T, 0, fr, cp, qc, nk, nv,
                                             lo, vmask, ok, O, lsum);
    } else {
        const uint k_begin = lo0 + s * args.split;
        verify_batches<false, KI>(kv, page_table, k_cache, v_cache, seq, g, k_begin, min(k_begin + args.split, p0), ki,
                                  fr, cp, qc, nk, nv, lo, vmask, ok, O, lsum);
    }

    // l over the 8 key rows (lane bits 1, 2, 4); O^T fragment i holds dims 16 r + i. Then the
    // KI simdgroups of each column group add up in tgm, where Q^T is no longer needed.
    lsum += simd_shuffle_xor(lsum, 2);
    lsum += simd_shuffle_xor(lsum, 4);
    lsum += simd_shuffle_xor(lsum, 16);
    const uint nv_cols = GROUP * T;  // the run's valid columns
    if (args.one_round) {
        // Every simdgroup's sums of the valid columns, [KI][5 T][128], then all threads add.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; e++) {
            const uint p = c * 8 + fcol + e;
            if (p >= nv_cols) continue;
            threadgroup float* os = tgm + (ki * nv_cols + p) * D + fr * 16;
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 16; i++) os[i] = O[i].thread_elements()[e];
            if (fr == 0) tls[ki * nv_cols + p] = lsum[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread j sums dim j % 128 of columns j / 128, + nsg / 4, ...
        const uint tid = sg * 32 + lane, d = tid % D;
        for (uint p = tid / D; p < nv_cols; p += nsg / 4) {
            float o = 0.0f, l = 0.0f;
            for (uint k = 0; k < KI; k++) {
                o += tgm[(k * nv_cols + p) * D + d];
                l += tls[k * nv_cols + p];
            }
            device float* pp = part + ((ulong(t0 + p / GROUP) * NQ + g * GROUP + p % GROUP) * args.max_splits + s) * PART;
            if (d == 0) pp[0] = l;
            pp[4 + d] = o;
        }
        return;
    }
    for (uint hw = KI / 2; hw >= 1; hw /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ki >= hw && ki < 2 * hw) {
            const uint slot = (ki - hw) * C + c;
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; e++) {
                const uint col = fcol + e;
                threadgroup float* os = tgm + (slot * 8 + col) * D + fr * 16;
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < 16; i++) os[i] = O[i].thread_elements()[e];
                if (fr == 0) tls[slot * 8 + col] = lsum[e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ki < hw) {
            const uint slot = ki * C + c;
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; e++) {
                const uint col = fcol + e;
                threadgroup const float* os = tgm + (slot * 8 + col) * D + fr * 16;
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < 16; i++) O[i].thread_elements()[e] += os[i];
                lsum[e] += tls[slot * 8 + col];
            }
        }
    }
    if (ki != 0) return;
    _Pragma("clang loop unroll(full)")
    for (uint e = 0; e < 2; e++) {
        const uint p = c * 8 + fcol + e;
        if (p >= nv_cols) continue;
        device float* pp = part + ((ulong(t0 + p / GROUP) * NQ + g * GROUP + p % GROUP) * args.max_splits + s) * PART;
        if (fr == 0) pp[0] = lsum[e];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 16; j += 4)
            *(device float4*)(pp + 4 + fr * 16 + j) = float4(O[j].thread_elements()[e], O[j + 1].thread_elements()[e],
                                                            O[j + 2].thread_elements()[e], O[j + 3].thread_elements()[e]);
    }
}

#define ATTN_VERIFY(KI) \
kernel void attn_verify_k##KI(device float* qkv [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const float* inv_rms [[buffer(2)]], device const uint2* meta [[buffer(3)]], device const uint* tokens [[buffer(4)]], \
    device const ushort* ve_table [[buffer(5)]], device const float* ve_gate [[buffer(6)]], \
    device const float* attn_gate [[buffer(7)]], device ushort* k_cache [[buffer(8)]], device ushort* v_cache [[buffer(9)]], \
    device const uint* page_table [[buffer(10)]], device const float2* rope_tab [[buffer(11)]], device float* part [[buffer(12)]], \
    device float* vdir [[buffer(13)]], device float* gates [[buffer(14)]], constant KvArgs& kv [[buffer(15)]], \
    constant AttnArgs& args [[buffer(16)]], device const uint2* runs [[buffer(17)]], device const uint* vis [[buffer(18)]], \
    device float* cap_q [[buffer(19)]], threadgroup float* tgm [[threadgroup(0)]], \
    threadgroup float* tls [[threadgroup(1)]], uint3 gid [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    attn_verify_impl<KI>(qkv, x, inv_rms, meta, tokens, ve_table, ve_gate, attn_gate, k_cache, v_cache, page_table, \
                         rope_tab, part, vdir, gates, kv, args, runs, vis, cap_q, gid, sg, nsg, lane, tgm, tls); \
}
ATTN_VERIFY(1)
ATTN_VERIFY(2)
ATTN_VERIFY(4)

// --------------------------------------------- chunk attention, matrix units
// For steps with chunks: one threadgroup per (tile of up to 8 consecutive tokens of one
// sequence, kv head, key split), one simdgroup per query head of the kv head, with the
// tile's tokens as the 8 columns. Keys before the tile are masked by each token's window,
// the tile's own by its `vis` (causal, or a draft tree's).
// K and V go into the MMAs as bf16, as they are in the cache: the matrix units multiply a
// bf16 operand by an fp32 one exactly. Their dims are permuted as load_batch_lines says.
// attn_sum_reduce sums the partials over all max_splits splits: splits past a tile's keys
// write zeros.
kernel void attn_mma(device const float* qkv [[buffer(0)]],
                     device const uint2* meta [[buffer(1)]],
                     device const ushort* k_cache [[buffer(2)]],
                     device const ushort* v_cache [[buffer(3)]],
                     device const uint* page_table [[buffer(4)]],
                     device float* part [[buffer(5)]],
                     device const uint2* tiles [[buffer(6)]],  // (first token, tokens <= 8)
                     constant KvArgs& kv [[buffer(7)]],
                     constant AttnArgs& args [[buffer(8)]],
                     device const uint* vis [[buffer(9)]],     // bit j: the tile's key j
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const uint g = gid.y, s = gid.z;
    const uint t0 = tiles[gid.x].x, nt = tiles[gid.x].y;
    const uint seq = meta[t0].x, p0 = meta[t0].y;
    // Keys of the tile's queries: [key_lo(p0), p0 + nt); this split's share of them.
    const uint k_begin = key_lo(kv, p0) + s * args.split;
    const uint k_end = min(k_begin + args.split, p0 + nt);
    const uint h = g * GROUP + sg;
    const uint2 fc = frag_coord(lane);
    const uint fr = fc.x, fcol = fc.y, cp = fcol / 2;

    // Q^T fragments (B operand): k-slot j of fragment f = dim_qk(f, j). Padding columns
    // repeat the last token, and are masked below.
    simdgroup_float8x8 QT[16];
    {
        device const float* qa = qkv + ulong(t0 + min(fcol, nt - 1)) * QKV + h * D;
        device const float* qb = qkv + ulong(t0 + min(fcol + 1, nt - 1)) * QKV + h * D;
        _Pragma("clang loop unroll(full)")
        for (uint f = 0; f < 16; f++) {
            const uint dim = dim_qk(f, fr);
            QT[f].thread_elements()[0] = qa[dim] * args.scale_log2;
            QT[f].thread_elements()[1] = qb[dim] * args.scale_log2;
        }
    }
    simdgroup_float8x8 O[16];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 16; i++) O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float2 lsum = 0.0f;
    // This lane's two columns: their first visible keys, and the tile's keys they see.
    const bool col_a = fcol < nt, col_b = fcol + 1 < nt;
    const uint lo_a = key_lo(kv, meta[t0 + min(fcol, nt - 1)].y), lo_b = key_lo(kv, meta[t0 + min(fcol + 1, nt - 1)].y);
    const uint vis_a = col_a ? vis[t0 + fcol] : 0u, vis_b = col_b ? vis[t0 + fcol + 1] : 0u;

    for (uint kb = k_begin; kb < k_end; kb += 8) {
        uint4 kw[4], vw[4];
        load_batch_lines(kv, page_table, k_cache, v_cache, seq, g, kb, k_end, fr, cp, kw, vw);
        simdgroup_float8x8 S[QK_CHAINS];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < QK_CHAINS; j++) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)")
        for (uint f = 0; f < 16; f++) {
            simdgroup_bfloat8x8 A;
            const bfloat2 w = as_type<bfloat2>(kw[f / 4][f % 4]);
            A.thread_elements()[0] = w[0];
            A.thread_elements()[1] = w[1];
            simdgroup_multiply_accumulate(S[f % QK_CHAINS], A, QT[f], S[f % QK_CHAINS]);
        }
        const float2 sc = qk_sum(S);
        const uint kp = kb + fr;  // this lane's key
        const bool kin = kp < k_end;  // then kp < p0 + nt
        const bool sa = kin && (kp < p0 ? kp >= lo_a : bool((vis_a >> (kp - p0)) & 1u));
        const bool sb = kin && (kp < p0 ? kp >= lo_b : bool((vis_b >> (kp - p0)) & 1u));
        const float pa = sa ? exp2(sc.x) : 0.0f;
        const float pb = sb ? exp2(sc.y) : 0.0f;
        lsum += float2(pa, pb);
        simdgroup_float8x8 P;
        P.thread_elements()[0] = pa;
        P.thread_elements()[1] = pb;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16; i++) {
            simdgroup_bfloat8x8 A;
            const bfloat2 wa = as_type<bfloat2>(vw[i / 8][(i % 8) / 2]), wb = as_type<bfloat2>(vw[2 + i / 8][(i % 8) / 2]);
            A.thread_elements()[0] = wa[i % 2];
            A.thread_elements()[1] = wb[i % 2];
            simdgroup_multiply_accumulate(O[i], A, P, O[i]);
        }
    }

    // l over the 8 key rows (lane bits 1, 2, 4); O^T fragment i's row r holds dim_o(i, r).
    lsum += simd_shuffle_xor(lsum, 2);
    lsum += simd_shuffle_xor(lsum, 4);
    lsum += simd_shuffle_xor(lsum, 16);
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < 2; c++) {
        if (fcol + c >= nt) continue;
        device float* pp = part + ((ulong(t0 + fcol + c) * NQ + h) * args.max_splits + s) * PART;
        if (fr == 0) pp[0] = lsum[c];
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16; i += 4)
            *(device float4*)(pp + 4 + dim_o(i, fr)) = float4(O[i].thread_elements()[c], O[i + 1].thread_elements()[c],
                                                             O[i + 2].thread_elements()[c], O[i + 3].thread_elements()[c]);
    }
}

// Sums attn_mma's splits, then XSA and the gate. One threadgroup per token, simdgroup =
// query head, lanes 4 dims each.
kernel void attn_sum_reduce(device const float* part [[buffer(0)]],
                            device const uint2* meta [[buffer(1)]],
                            device const float* vdir [[buffer(2)]],
                            device const float* gates [[buffer(3)]],
                            device const float* xsa [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            device float* sdpa [[buffer(6)]],
                            constant KvArgs& kv [[buffer(7)]],
                            constant AttnArgs& args [[buffer(8)]],
                            uint t [[threadgroup_position_in_grid]],
                            uint h [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    device const float* pp = part + (ulong(t) * NQ + h) * args.max_splits * PART;
    float L = 0.0f;
    float4 O = 0.0f;
    for (uint s0 = 0; s0 < args.fixed_splits; s0 += 4) {
        float l4[4];
        float4 o4[4];
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 4; i++) {
            const uint si = min(s0 + i, args.fixed_splits - 1);
            l4[i] = pp[si * PART];
            o4[i] = *(device const float4*)(pp + si * PART + 4 + lane * 4);
        }
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 4; i++) {
            if (s0 + i < args.fixed_splits) { L += l4[i]; O += o4[i]; }
        }
    }
    float4 o = O / L;
    const ulong oi = ulong(t) * H + h * D + lane * 4;
    if (args.capture) *(device float4*)(sdpa + oi) = o;
    float4 vd = *(device const float4*)(vdir + (ulong(t) * NKV + h / GROUP) * D + lane * 4);
    float proj = simd_sum(dot(o, vd));
    o -= xsa[h] * proj * vd;
    *(device float4*)(out + oi) = o * gates[t * NQ + h];
}

// Sums attn_verify's splits, then XSA and the gate. One threadgroup of 512 per (token,
// head): four groups of 128 threads (thread = dim), group j taking splits j, j + 4, ...
kernel void attn_dec_reduce(device const float* part [[buffer(0)]],
                            device const uint2* meta [[buffer(1)]],
                            device const float* vdir [[buffer(2)]],
                            device const float* gates [[buffer(3)]],
                            device const float* xsa [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            device float* sdpa [[buffer(6)]],
                            constant KvArgs& kv [[buffer(7)]],
                            constant AttnArgs& args [[buffer(8)]],
                            device const uint2* tin [[buffer(9)]],
                            uint2 gid [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const uint t = gid.x, h = gid.y;
    const uint d = tid % D, grp = tid / D;
    const uint p0 = meta[t - tin[t].y].y;  // the run's first position, which sets its splits
    const uint nsplit = (p0 - key_lo(kv, p0) + args.split - 1) / args.split + 1;
    device const float* pp = part + (ulong(t) * NQ + h) * args.max_splits * PART;
    float L = 0.0f, O = 0.0f;
    for (uint s0 = grp; s0 < nsplit; s0 += 32) {
        float l8[8], o8[8];
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 8; i++) {
            const uint si = min(s0 + 4 * i, nsplit - 1);
            l8[i] = pp[si * PART];
            o8[i] = pp[si * PART + 4 + d];
        }
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 8; i++) {
            if (s0 + 4 * i < nsplit) { L += l8[i]; O += o8[i]; }
        }
    }
    threadgroup float ls[4], os[4][D], red[4];
    if (d == 0) ls[grp] = L;
    os[grp][d] = O;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (grp != 0) return;
    const float vd = vdir[(ulong(t) * NKV + h / GROUP) * D + d];
    const float o = (os[0][d] + os[1][d] + os[2][d] + os[3][d]) / (ls[0] + ls[1] + ls[2] + ls[3]);
    const float pr = simd_sum(o * vd);
    if (lane == 0) red[sg] = pr;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong oi = ulong(t) * H + h * D + d;
    if (args.capture) sdpa[oi] = o;
    out[oi] = (o - xsa[h] * (red[0] + red[1] + red[2] + red[3]) * vd) * gates[t * NQ + h];
}

// --------------------------------------------------------------------- MUDD
// At layers 24 and 47: one inner projection of rms_norm(current) feeds two mixes of the
// taps (hist[a], hist[b], current), the attention input and the residual base. One
// threadgroup of 256 per token.
kernel void mudd(device const float* cur [[buffer(0)]],
                 device const float* tap0 [[buffer(1)]],
                 device const float* tap1 [[buffer(2)]],
                 device const float* dense1 [[buffer(3)]],   // [32][1280]
                 device const float* mix [[buffer(4)]],      // [2][3][33] for this site
                 device float* out_attn [[buffer(5)]],
                 device float* out_base [[buffer(6)]],
                 uint t [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float inner[32];
    threadgroup float w[6];
    device const float* c = cur + ulong(t) * H;
    float ss = 0.0f;
    for (uint i = 0; i < 5; i++) { float v = c[tid + 256 * i]; ss += v * v; }
    const float inv = rsqrt(tg_sum(ss, red, sg, lane, 8) / float(H) + RMS_EPS);
    for (uint j = sg * 4; j < sg * 4 + 4; j++) {
        float s = 0.0f;
        for (uint k = lane * 4; k < H; k += 128)
            s += dot(*(device const float4*)(c + k) * inv, *(device const float4*)(dense1 + j * H + k));
        s = simd_sum(s);
        if (lane == 0) inner[j] = gelu(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const float* row = mix + tid * 33;
        float s = 0.0f;
        for (uint j = 0; j < 32; j++) s += inner[j] * row[j];
        w[tid] = s + row[32];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 5; i++) {
        ulong o = ulong(t) * H + tid + 256 * i;
        float a = tap0[o], b = tap1[o], x = c[tid + 256 * i];
        out_attn[o] = w[0] * a + w[1] * b + w[2] * x;
        out_base[o] = w[3] * a + w[4] * b + w[5] * x;
    }
}

// ------------------------------------------------------------------- sampling
// argmax_partial reduces blocks of [rows][vocab] logits to their max and argmax. `sample`
// then picks each row's token: the argmax at temperature 0 (ties to the lowest id), else a
// draw from the top-p set, as HF's TemperatureLogitsWarper then TopPLogitsWarper.
constant constexpr uint AM_BLOCK = 4096;
constant constexpr uint NO_TOK = 0xffffffffu;  // SampleRow: no token of the step
constant constexpr uint SAMPLE_THREADS = 1024;
constant constexpr uint SAMPLE_ROUNDS = 8;

inline void better(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}
inline void simd_best(thread float& bv, thread uint& bi) {
    for (ushort o = 16; o > 0; o >>= 1) {
        float v = simd_shuffle_xor(bv, o);
        uint i = simd_shuffle_xor(bi, o);
        better(bv, bi, v, i);
    }
}

// grid: (ceil(vocab / 4096), rows), 256 threads
kernel void argmax_partial(device const float* logits [[buffer(0)]],
                           device float2* part [[buffer(1)]],   // (value, as_type<float>(index))
                           constant uint& vocab [[buffer(2)]],
                           uint2 gid [[threadgroup_position_in_grid]],
                           uint2 ngrid [[threadgroups_per_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bvs[8];
    threadgroup uint bis[8];
    device const float* row = logits + ulong(gid.y) * vocab;
    float bv = -FLT_MAX;
    uint bi = 0xffffffffu;
    for (uint i = gid.x * AM_BLOCK + tid; i < min((gid.x + 1) * AM_BLOCK, vocab); i += 256) better(bv, bi, row[i], i);
    simd_best(bv, bi);
    if (lane == 0) { bvs[sg] = bv; bis[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint i = 1; i < 8; i++) better(bv, bi, bvs[i], bis[i]);
        part[gid.y * ngrid.x + gid.x] = float2(bv, as_type<float>(bi));
    }
}

// One head row's sampling settings (engine::Sampling), mirrored in kernels.rs.
struct SampleRow {
    uint seed_lo, seed_hi;
    uint pos;           // position of the token drawn, if tok is NO_TOK
    float temperature;  // 0: argmax
    float top_p;
    uint tok;           // the step's token whose head row this is: the draw is at its position + 1
};

// Philox4x32-10 (Salmon et al. 2011, Random123's constants): a counter-based generator, so
// any draw is a pure function of (key, counter).
inline uint4 philox(uint4 c, uint2 k) {
    for (uint i = 0; i < 10; i++) {
        uint hi0 = mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        uint hi1 = mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

// Uniform in [0, 1) on a 2^-24 grid, for rejection round `round` of the draw at `pos`.
inline float uniform01(SampleRow row, uint pos, uint round) {
    return float(philox(uint4(pos, round, 0, 0), uint2(row.seed_lo, row.seed_hi)).x >> 8) * 0x1p-24f;
}

// Index of thread t's prefix sum in `ex`, padded against bank conflicts.
inline uint ex_at(uint t) { return t + t / 32; }

// ex[ex_at(t)] = the sum of v over threads before t, ex[ex_at(1024)] = the total. Summed
// in thread order, one addition at a time, so the sums never decrease along the order.
inline void tg_scan(float v, threadgroup float* ex, threadgroup float* base, uint tid) {
    ex[ex_at(tid)] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float acc = 0.0f;
        for (uint i = 32 * tid; i < 32 * tid + 32; i++) { float x = ex[ex_at(i)]; ex[ex_at(i)] = acc; acc += x; }
        base[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float acc = 0.0f;
        for (uint g = 0; g < 32; g++) { float x = base[g]; base[g] = acc; acc += x; }
        ex[ex_at(SAMPLE_THREADS)] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ex[ex_at(tid)] += base[tid / 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Each row's token, into `out`. grid: rows threadgroups of 1024 threads.
//
// The draw needs no sort. With e_i = exp((x_i - max) / T), it inverts the CDF of e over a
// fixed order (thread t's ids t, t + 1024, ..., threads in turn) at u * sum(e), and accepts
// the token j if the tokens more likely than j hold less than top_p of the mass. Otherwise
// it draws again from those tokens alone, which still hold every top-p token; after
// SAMPLE_ROUNDS, the argmax.
kernel void sample(device const float* logits [[buffer(0)]],
                   device const float2* part [[buffer(1)]],
                   device const SampleRow* rows [[buffer(2)]],
                   device uint* out [[buffer(3)]],
                   device const uint2* meta [[buffer(4)]],
                   constant uint& nblocks [[buffer(5)]],
                   constant uint& vocab [[buffer(6)]],
                   uint r [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ex[SAMPLE_THREADS + SAMPLE_THREADS / 32 + 1];
    threadgroup float base[32];
    threadgroup uint pick;
    const SampleRow row = rows[r];
    const uint pos = row.tok == NO_TOK ? row.pos : meta[row.tok].y + 1;
    float lmax = -FLT_MAX;
    uint tok = 0xffffffffu;
    for (uint i = lane; i < nblocks; i += 32) {
        float2 p = part[r * nblocks + i];
        better(lmax, tok, p.x, as_type<uint>(p.y));
    }
    simd_best(lmax, tok);
    if (row.temperature > 0.0f) {
        device const float* x = logits + ulong(r) * vocab;
        const float c = M_LOG2E_F / row.temperature;
        float s = 0.0f;
        for (uint i = tid; i < vocab; i += SAMPLE_THREADS) s += exp2((x[i] - lmax) * c);
        tg_scan(s, ex, base, tid);
        const float z = ex[ex_at(SAMPLE_THREADS)];
        float low = -INFINITY;  // the current round draws from the tokens above this logit
        for (uint round = 0; round < SAMPLE_ROUNDS; round++) {
            const float target = uniform01(row, pos, round) * ex[ex_at(SAMPLE_THREADS)];
            if (tid == 0) pick = 0xffffffffu;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float lo = ex[ex_at(tid)];
            if (lo <= target && target < ex[ex_at(tid + 1)]) {
                // Re-adds this thread's terms in the order of its sum. Rounding can leave
                // the target past them; then the last one counts.
                float acc = 0.0f;
                uint j = 0xffffffffu;
                for (uint i = tid; i < vocab; i += SAMPLE_THREADS) {
                    float xi = x[i];
                    if (!(xi > low)) continue;
                    float e = exp2((xi - lmax) * c);
                    acc += e;
                    if (e > 0.0f) j = i;
                    if (target - lo < acc) break;
                }
                pick = j;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint j = pick;
            if (j == 0xffffffffu) break;
            const float xj = x[j];
            if (xj == lmax || row.top_p >= 1.0f) { tok = j; break; }
            float q = 0.0f;
            for (uint i = tid; i < vocab; i += SAMPLE_THREADS) {
                float xi = x[i];
                if (xi > xj) q += exp2((xi - lmax) * c);
            }
            tg_scan(q, ex, base, tid);
            if (ex[ex_at(SAMPLE_THREADS)] < row.top_p * z) { tok = j; break; }
            low = xj;
        }
    }
    if (tid == 0) out[r] = tok;
}

// ----------------------------------------------------------------- accepting
// After `sample`: what each run emits, and how far its sequence advances.
//   - A run of CPU tokens (a prompt chunk) advances by its length, and emits its last head
//     row's token, if it has one.
//   - A verify run has the head on every row, and row i's token is the one decoding one at
//     a time would draw after row i's input. From the root, the path follows the child
//     whose draft is the token drawn on its parent's row; the run emits the path's head
//     tokens, moved to the front of its `out`. The path's rows go to path[run] for
//     kv_compact.
// The emitted tokens stop after a stop token and at the budget, which ends the slot's
// generation: later steps of a done slot emit nothing. The last emitted token becomes the
// slot's next input. on_path[t] says whether token t is now in its sequence.
struct RunIn {
    uint slot, first, count;  // the run's tokens in the step
    uint head0, nhead;        // its head rows
    uint verify;
};
constant constexpr uint MAX_STOP = 8;
struct GenState {
    uint remaining;  // tokens it may still emit
    uint done;
    uint nstop;
    uint pad;
    uint stop[MAX_STOP];
};

// grid: ceil(runs / 64) threadgroups of 64 threads, one per run.
kernel void accept(device const RunIn* runs [[buffer(0)]],
                   device const uint* tokens [[buffer(1)]],  // the step's inputs, resolved by embed
                   device uint* out [[buffer(2)]],           // the head rows' tokens
                   device uint* lens [[buffer(3)]],
                   device uint* feed [[buffer(4)]],
                   device GenState* gen [[buffer(5)]],
                   device uint* emitted [[buffer(6)]],       // per run: tokens emitted
                   constant uint& nruns [[buffer(7)]],
                   device const uint* tree [[buffer(8)]],
                   device uint* path [[buffer(9)]],          // per run: [FEED_W] rows
                   device uint* path_len [[buffer(10)]],     // per run
                   device uint* on_path [[buffer(11)]],      // per token
                   uint r [[thread_position_in_grid]]) {
    if (r >= nruns) return;
    const RunIn ru = runs[r];
    device GenState& gs = gen[ru.slot];
    path_len[r] = 0;
    for (uint j = 0; j < ru.count; j++) on_path[ru.first + j] = !ru.verify && !gs.done;
    if (gs.done) { emitted[r] = 0; return; }
    if (!ru.verify) lens[ru.slot] += ru.count;
    if (ru.nhead == 0) { emitted[r] = 0; return; }
    uint first = ru.head0 + ru.nhead - 1, e = 1;
    uint walk[FEED_W];
    walk[0] = 0;
    if (ru.verify) {
        // Children come after their parents, so one pass over the rows finds the path.
        device const uint* node = tree + ru.slot * FEED_W;
        for (uint c = 1, at = 0; c < ru.count; c++) {
            if (node_parent(node[c]) == at && tokens[ru.first + c] == out[ru.head0 + at]) walk[e++] = at = c;
        }
        for (uint j = 1; j < e; j++) out[ru.head0 + j] = out[ru.head0 + walk[j]];  // walk[j] >= j
        first = ru.head0;
    }
    bool stop = false;
    uint k = 0;
    while (k < e && k < gs.remaining && !stop) {
        const uint tok = out[first + k];
        for (uint j = 0; j < gs.nstop; j++) stop = stop || tok == gs.stop[j];
        k++;
    }
    if (ru.verify) {
        lens[ru.slot] += k;
        path_len[r] = k;
        for (uint j = 0; j < k; j++) {
            path[r * FEED_W + j] = walk[j];
            on_path[ru.first + walk[j]] = 1;
        }
    }
    feed[ru.slot * FEED_W] = out[first + k - 1];
    gs.remaining -= k;
    if (stop || gs.remaining == 0) gs.done = 1;
    emitted[r] = k;
}

// After `accept`: the K/V of each verify run's path, on every layer, from p0 + path[j] to
// p0 + j (p0 the run's first position). path[j] >= j and increases, so nothing is
// overwritten before it's read. grid: (runs, KV layers: the local ones, then the global
// ones), 64 threads (simdgroup = kv head, lane = 4 dims).
kernel void kv_compact(device const RunIn* runs [[buffer(0)]],
                       device const uint* path [[buffer(1)]],
                       device const uint* path_len [[buffer(2)]],
                       device const uint* lens [[buffer(3)]],  // as accept left them
                       device ushort* k_local [[buffer(4)]],
                       device ushort* v_local [[buffer(5)]],
                       device ushort* k_global [[buffer(6)]],
                       device ushort* v_global [[buffer(7)]],
                       device const uint* page_table [[buffer(8)]],
                       constant KvArgs& local [[buffer(9)]],
                       constant KvArgs& global [[buffer(10)]],
                       constant uint& n_local [[buffer(11)]],
                       uint2 gid [[threadgroup_position_in_grid]],
                       uint g [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r = gid.x, l = gid.y, k = path_len[r];
    const uint slot = runs[r].slot, p0 = lens[slot] - k;
    const bool gl = l >= n_local;
    constant KvArgs& kv = gl ? global : local;
    device ushort* kc = gl ? k_global : k_local;
    device ushort* vc = gl ? v_global : v_local;
    const uint layer = gl ? l - n_local : l;
    for (uint j = 1; j < k; j++) {
        const uint from = path[r * FEED_W + j];
        if (from == j) continue;
        const ulong src = kv_row_of(kv, layer, page_table, slot, g, p0 + from) + lane * 4;
        const ulong dst = kv_row_of(kv, layer, page_table, slot, g, p0 + j) + lane * 4;
        *(device ushort4*)(kc + dst) = *(device const ushort4*)(kc + src);
        *(device ushort4*)(vc + dst) = *(device const ushort4*)(vc + src);
    }
}

// A test drafter: the drafts of slot s are the tokens of a recorded sequence (trace[s], by
// position), wrong at the positions a hash of (seed, position) puts under `rate`. After
// `accept`, for every run with head rows. grid: runs threadgroups of FEED_W threads.
struct ReplayArgs {
    uint stride;  // trace row length (max_seq_len)
    uint seed;
    float rate;
    uint vocab;
};

kernel void replay_draft(device const RunIn* runs [[buffer(0)]],
                         device const uint* lens [[buffer(1)]],
                         device uint* feed [[buffer(2)]],
                         device const uint* trace [[buffer(3)]],
                         device const uint* trace_len [[buffer(4)]],
                         constant ReplayArgs& ra [[buffer(5)]],
                         constant uint& nruns [[buffer(6)]],
                         device uint* tree [[buffer(7)]],
                         uint r [[threadgroup_position_in_grid]],
                         uint i [[thread_index_in_threadgroup]]) {
    if (i == 0 || i >= FEED_W || r >= nruns || runs[r].nhead == 0) return;
    const uint slot = runs[r].slot, n = trace_len[slot];
    if (n == 0) return;
    const uint pos = lens[slot] + i;
    uint tok = pos < n ? trace[ulong(slot) * ra.stride + pos] : 0u;
    const uint h = philox(uint4(pos, 0x5eed, 0, 0), uint2(ra.seed, 0x0dd5u)).x;
    if (float(h >> 8) * 0x1p-24f < ra.rate) tok = (tok + 1) % ra.vocab;
    feed[slot * FEED_W + i] = tok;
    tree[slot * FEED_W + i] = chain_node(i);
}

// ------------------------------------------------------------------- drafter
// The drafter (engine/src/engine/drafter.rs, tools/spec/drafter.py) runs on the target's
// kernels; these are its own pieces.

// Context keys and values: kvp[t] = [layer][k | v][2 heads][128], the fused context through
// every draft layer's k/v weights. Norms and ropes k, and writes both to the drafter's ring
// at the token's position, a ring per (layer, slot) as the target's local layers'. Only the
// tokens on_path: a draft tree's other rows share their positions. grid: (tokens, layers),
// 64 threads (simdgroup = kv head).
kernel void draft_kv(device const float* kvp [[buffer(0)]],
                     device const uint2* meta [[buffer(1)]],
                     device const float2* rope_tab [[buffer(2)]],
                     device ushort* k_ring [[buffer(3)]],
                     device ushort* v_ring [[buffer(4)]],
                     constant KvArgs& kv [[buffer(5)]],
                     device const uint* on_path [[buffer(6)]],
                     uint2 gid [[threadgroup_position_in_grid]],
                     uint2 ngrid [[threadgroups_per_grid]],
                     uint g [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const uint t = gid.x, l = gid.y;
    if (!on_path[t]) return;
    const uint seq = meta[t].x, pos = meta[t].y;
    const uint d0 = lane * 4;
    device const float* row = kvp + (ulong(t) * ngrid.y + l) * (2 * NKV * D);
    const float4 k = norm_rope(*(device const float4*)(row + g * D + d0), true, lane, rope_tab + ulong(pos) * 32);
    const float4 v = *(device const float4*)(row + NKV * D + g * D + d0);
    const ulong r = ((ulong(l) * kv.max_seqs + seq) * NKV + g) * kv.ring * D + ulong(pos % kv.ring) * D + d0;
    *(device ushort4*)(k_ring + r) = ushort4(to_bf(k.x), to_bf(k.y), to_bf(k.z), to_bf(k.w));
    *(device ushort4*)(v_ring + r) = ushort4(to_bf(v.x), to_bf(v.y), to_bf(v.z), to_bf(v.w));
}

// The normed fused context of the step's tokens on_path, into a ring per slot of cr
// positions, for the slot's next block to read at its position - 1. One threadgroup of 256
// per token.
kernel void draft_ctx(device const float* ctx [[buffer(0)]],
                      device const float* ctx_inv [[buffer(1)]],
                      device const uint2* meta [[buffer(2)]],
                      device float* ring [[buffer(3)]],
                      constant uint& cr [[buffer(4)]],
                      device const uint* on_path [[buffer(5)]],
                      uint t [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    if (!on_path[t]) return;
    const uint seq = meta[t].x, pos = meta[t].y;
    device float* dst = ring + (ulong(seq) * cr + pos % cr) * H;
    for (uint j = 0; j < 5; j++) dst[tid + 256 * j] = ctx[ulong(t) * H + tid + 256 * j] * ctx_inv[t];
}

// The block's rows. Row i of a slot's block sits at position lens[slot] + i: row 0 is the
// slot's next input, its bonus token, as rms_norm(embedding); the rows after it hold the
// mask embedding. With cr > 0, each row also gets the normed context at lens[slot] - 1 from
// draft_ctx's ring, into `last`. One threadgroup of 256 per row.
kernel void draft_embed(device const ushort* table [[buffer(0)]],
                        device const uint2* tin [[buffer(1)]],
                        device const uint* lens [[buffer(2)]],
                        device const uint* feed [[buffer(3)]],
                        device const float* mask_emb [[buffer(4)]],
                        device float* x [[buffer(5)]],
                        device uint2* meta [[buffer(6)]],
                        device uint* tokens [[buffer(7)]],
                        device const float* ring [[buffer(8)]],
                        device float* last [[buffer(9)]],
                        constant uint& cr [[buffer(10)]],
                        uint t [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    const uint slot = tin[t].x, i = tin[t].y;
    const uint tok = feed[slot * FEED_W];
    if (tid == 0) {
        meta[t] = uint2(slot, lens[slot] + i);
        tokens[t] = tok;
    }
    if (cr > 0) {
        device const float* src = ring + (ulong(slot) * cr + (lens[slot] - 1) % cr) * H;
        for (uint j = 0; j < 5; j++) last[ulong(t) * H + tid + 256 * j] = src[tid + 256 * j];
    }
    if (i > 0) {
        for (uint j = 0; j < 5; j++) x[ulong(t) * H + tid + 256 * j] = mask_emb[tid + 256 * j];
        return;
    }
    device const ushort* row = table + ulong(tok) * H;
    float v[5];
    float ss = 0.0f;
    for (uint j = 0; j < 5; j++) { v[j] = bf(row[tid + 256 * j]); ss += v[j] * v[j]; }
    const float inv = rsqrt(tg_sum(ss, red, sg, lane, 8) / float(H) + RMS_EPS);
    for (uint j = 0; j < 5; j++) x[ulong(t) * H + tid + 256 * j] = v[j] * inv;
}

// DFlash-2's two-tap convolution within each block (tools/spec/drafter.py, Conv2): out[t] =
// resid[t] (if has_resid) + k0 * x[t] + k1 * x[t - 1], with no predecessor for a block's row
// 0. The kernels are base [2][H] plus corr [rows][CONV_KC] per 16 channels (k0's
// corrections, then k1's). out mustn't be x. One threadgroup of 256 per row.
constant constexpr uint CONV_KC = 192;
kernel void draft_conv(device const float* x [[buffer(0)]],
                       device const float* corr [[buffer(1)]],
                       device const float* base [[buffer(2)]],
                       device const uint2* tin [[buffer(3)]],
                       device const float* resid [[buffer(4)]],
                       device float* out [[buffer(5)]],
                       constant uint& has_resid [[buffer(6)]],
                       uint t [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    const bool prev = tin[t].y > 0;
    device const float* c = corr + ulong(t) * CONV_KC;
    for (uint j = 0; j < 5; j++) {
        const uint d = tid + 256 * j;
        const ulong at = ulong(t) * H + d;
        float y = (base[d] + c[d / 16]) * x[at];
        if (prev) y += (base[H + d] + c[H / 16 + d / 16]) * x[at - H];
        out[at] = has_resid ? resid[at] + y : y;
    }
}

// Each drafted block row's `ncand` best tokens over the draft vocabulary, as ids, and their
// probabilities at temperature 1 / inv_t, for draft_tree. Row 0, the bonus token's own, is
// skipped. One threadgroup of 1024 per row.
constant constexpr uint CANDS = 4;  // per row, at most
kernel void draft_top(device const float* logits [[buffer(0)]],
                      device const uint* vocab [[buffer(1)]],
                      device const uint2* tin [[buffer(2)]],
                      device uint* cand [[buffer(3)]],     // [row][CANDS]
                      device float* cand_q [[buffer(4)]],  // [row][CANDS]
                      constant uint& nv [[buffer(5)]],
                      constant float& inv_t [[buffer(6)]],
                      constant uint& ncand [[buffer(7)]],  // <= CANDS
                      uint r [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bvs[32];
    threadgroup uint bis[32];
    threadgroup float top_v[CANDS];
    threadgroup uint top_i[CANDS];
    if (tin[r].y == 0) return;
    device const float* row = logits + ulong(r) * nv;
    for (uint c = 0; c < ncand; c++) {
        float bv = -FLT_MAX;
        uint bi = 0xffffffffu;
        for (uint j = tid; j < nv; j += 1024) {
            bool taken = false;
            for (uint m = 0; m < c; m++) taken = taken || top_i[m] == j;
            if (!taken) better(bv, bi, row[j], j);
        }
        simd_best(bv, bi);
        if (lane == 0) { bvs[sg] = bv; bis[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint j = 1; j < 32; j++) better(bv, bi, bvs[j], bis[j]);
            top_v[c] = bv;
            top_i[c] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float z = 0.0f;
    for (uint j = tid; j < nv; j += 1024) z += exp((row[j] - top_v[0]) * inv_t);
    z = tg_sum(z, bvs, sg, lane, 32);
    if (tid < ncand) {
        cand[r * CANDS + tid] = vocab[top_i[tid]];
        cand_q[r * CANDS + tid] = exp((top_v[tid] - top_v[0]) * inv_t) / z;
    }
}

// Each block's draft tree, best-first over its rows' `ncand` candidates (DDTree, arXiv
// 2604.12989): the next row is the best child not yet taken of any row so far, a node's
// score being the product of q along its path. Block row d's candidates are the children at
// depth d of any parent: the rows are marginals. The tree goes to feed[slot][1..] and
// tree[slot][1..], parents first, and the rows past it are dead. One thread per block.
kernel void draft_tree(device const uint2* blocks [[buffer(0)]],  // (first row, rows)
                       device const uint2* tin [[buffer(1)]],
                       device const uint* cand [[buffer(2)]],
                       device const float* cand_q [[buffer(3)]],
                       device uint* feed [[buffer(4)]],
                       device uint* tree [[buffer(5)]],
                       constant uint& nblocks [[buffer(6)]],
                       constant uint& ncand [[buffer(7)]],
                       uint b [[thread_position_in_grid]]) {
    if (b >= nblocks) return;
    const uint first = blocks[b].x, rows = blocks[b].y, slot = tin[first].x;
    // Per row so far: its path's score, its depth, its candidates taken as children, and the
    // rows it sees. Row 0 is the root.
    float score[VERIFY_T] = {1.0f};
    uint depth[VERIFY_T] = {0}, taken[VERIFY_T] = {0}, sees[VERIFY_T] = {1};
    uint n = 1;
    for (; n < rows; n++) {
        float best = -1.0f;
        uint p = 0;
        for (uint i = 0; i < n; i++) {
            if (depth[i] + 1 == rows || taken[i] == ncand) continue;
            const float s = score[i] * cand_q[(first + depth[i] + 1) * CANDS + taken[i]];
            if (s > best) {
                best = s;
                p = i;
            }
        }
        if (best < 0.0f) break;
        const uint d = depth[p] + 1;
        feed[slot * FEED_W + n] = cand[(first + d) * CANDS + taken[p]];
        taken[p]++;
        score[n] = best;
        depth[n] = d;
        taken[n] = 0;
        sees[n] = sees[p] | 1u << n;
        tree[slot * FEED_W + n] = tree_node(d, p, sees[n]);
    }
    for (; n < FEED_W; n++) tree[slot * FEED_W + n] = dead_node(n);
}
