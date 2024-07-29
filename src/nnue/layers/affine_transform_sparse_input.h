/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of layer AffineTransformSparseInput of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

#include "../../bitboard.h"
#include "../nnue_common.h"
#include "affine_transform.h"
#include "simd.h"

/*
  This file contains the definition for the activation of both accumulators, followed by a fully connected layer (aka affine transform) with block sparse input.
*/

namespace Stockfish::Eval::NNUE::Layers {

#if (USE_SSSE3 | (USE_NEON >= 8))
alignas(CacheLineSize) static inline const
  std::array<std::array<std::uint16_t, 8>, 32768> lookup_indices = []() {
      std::array<std::array<std::uint16_t, 8>, 32768> v{};
      for (unsigned i = 0; i < 32768; ++i)
      {
          std::uint64_t j = i, k = 0;
          while (j)
              v[i][k++] = pop_lsb(j);
      }
      return v;
  }();
#endif

// Sparse input implementation
template<IndexType InDims, IndexType OutDims>
class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType  = std::int16_t;
    using OutputType = std::int32_t;
    using AffineType = std::uint8_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static_assert(OutputDimensions % 16 == 0,
                  "Only implemented for OutputDimensions divisible by 16.");

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

#if (USE_SSSE3 | (USE_NEON >= 8))
    static constexpr IndexType ChunkSize = 4;
#else
    static constexpr IndexType ChunkSize = 1;
#endif

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0xCC03DAE4u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / ChunkSize) % (PaddedInputDimensions / ChunkSize) * OutputDimensions * ChunkSize
             + i / PaddedInputDimensions * ChunkSize + i % ChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if (USE_SSSE3 | (USE_NEON >= 8))
        return get_weight_index_scrambled(i);
#else
        return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);

        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

        return !stream.fail();
    }
    // Forward propagation
    void propagate(const InputType* us, const InputType* them, OutputType* output) const {

#if (USE_SSSE3 | (USE_NEON >= 8))
    #if defined(USE_AVX512)
        using invec_t  = __m512i;
        using u8vec_t  = __m512i;
        using u32vec_t = __m512i;
        using outvec_t = __m512i;
        #define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
        #define vec_zero() _mm512_setzero_epi32()
        #define vec_set_16(a) _mm512_set1_epi16(a)
        #define vec_max_16(a, b) _mm512_max_epi16(a, b)
        #define vec_min_16(a, b) _mm512_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
        #define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
        #define vec_broadcast_nth_u32(a, n) _mm512_permutexvar_epi32(_mm512_set1_epi32(n), a)
        #define vec_add_dpbusd_32 Simd::m512_add_dpbusd_epi32
        #define vec_nnz(a) _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512())
    #elif defined(USE_AVX2)
        using invec_t  = __m256i;
        using u8vec_t  = __m256i;
        using u32vec_t = __m256i;
        using outvec_t = __m256i;
        #define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
        #define vec_zero() _mm256_setzero_si256()
        #define vec_set_16(a) _mm256_set1_epi16(a)
        #define vec_max_16(a, b) _mm256_max_epi16(a, b)
        #define vec_min_16(a, b) _mm256_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
        #define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
        #define vec_broadcast_nth_u32(a, n) _mm256_permutevar8x32_epi32(_mm256_set1_epi32(n), a)
        #define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32
        #if defined(USE_VNNI) && !defined(USE_AVXVNNI)
            #define vec_nnz(a) _mm256_cmpgt_epi32_mask(a, _mm256_setzero_si256())
        #else
            #define vec_nnz(a) \
                _mm256_movemask_ps( \
                  _mm256_castsi256_ps(_mm256_cmpgt_epi32(a, _mm256_setzero_si256())))
        #endif
    #elif defined(USE_SSSE3)
        using invec_t  = __m128i;
        using u8vec_t  = __m128i;
        using u32vec_t = __m128i;
        using outvec_t = __m128i;
        #define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
        #define vec_zero() _mm_setzero_si128()
        #define vec_set_16(a) _mm_set1_epi16(a)
        #define vec_max_16(a, b) _mm_max_epi16(a, b)
        #define vec_min_16(a, b) _mm_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm_slli_epi16(a, b)
        #define vec_packus_16(a, b) _mm_packus_epi16(a, b)
        #define vec_broadcast_nth_u32(a, n) _mm_cvtsi128_si32(_mm_bsrli_si128(a, n * 4))
        #define vec_add_dpbusd_32 Simd::m128_add_dpbusd_epi32
        #define vec_nnz(a) \
            _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128())))
    #elif defined(USE_NEON_DOTPROD)
        using invec_t  = int16x8_t;
        using u8vec_t  = int8x16_t;
        using u32vec_t = uint32x4_t;
        using outvec_t = int32x4_t;
        #define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
        #define vec_zero() invec_t { 0 }
        #define vec_set_16(a) vdupq_n_s16(a)
        #define vec_max_16(a, b) vmaxq_s16(a, b)
        #define vec_min_16(a, b) vminq_s16(a, b)
        #define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
        #define vec_packus_16(a, b) vcombine_u8(vqmovun_s16(a), vqmovun_s16(b))
        #define vec_broadcast_nth_u32(a, n) vreinterpretq_s8_u32(vdupq_n_u32(vgetq_lane_s32(a, n)))
        #define vec_add_dpbusd_32 Simd::dotprod_m128_add_dpbusd_epi32
        #define vec_nnz(a) vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)))
    #elif defined(USE_NEON)
        using invec_t  = int8x16_t;
        using u8vec_t  = int8x16_t;
        using u32vec_t = uint32x4_t;
        using outvec_t = int32x4_t;
        #define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
        #define vec_zero() invec_t { 0 }
        #define vec_set_16(a) vdupq_n_s16(a)
        #define vec_max_16(a, b) vmaxq_s16(a, b)
        #define vec_min_16(a, b) vminq_s16(a, b)
        #define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
        #define vec_packus_16(a, b) vcombine_u8(vqmovun_s16(a), vqmovun_s16(b))
        #define vec_broadcast_nth_u32(a, n) vreinterpretq_s8_u32(vdupq_n_u32(vgetq_lane_s32(a, n)))
        #define vec_add_dpbusd_32 Simd::neon_m128_add_dpbusd_epi32
        #define vec_nnz(a) vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)))
    #endif
        static constexpr IndexType OutputSimdWidth = sizeof(outvec_t) / sizeof(OutputType);
        static_assert(InputDimensions % 128 == 0);

        constexpr IndexType NumChunks = InputDimensions / 2 / sizeof(u8vec_t);
        constexpr IndexType NumRegs   = OutputDimensions / OutputSimdWidth;
        IndexType           count;

        const outvec_t* biasvec = reinterpret_cast<const outvec_t*>(biases);
        outvec_t        sum[NumRegs];
        for (IndexType k = 0; k < NumRegs; ++k)
            sum[k] = biasvec[k];

        const invec_t Zero = vec_zero();
        const invec_t One  = vec_set_16(127 * 2);
        IndexType offset = 0;
        for (InputType* acc : {us, them}) {
            const invec_t* in0 = &acc[0];
            const invec_t* in1 = &acc[InputDimensions / 2];
            for (IndexType i = 0; i < NumChunks; ++i) {

                // First, we perform ClippedReLU and pairwise multiplication to activate the inputs (accumulators).
                // Then, we will perform affine transform in a sparse manner.

                // What we want to do is multiply inputs in a pairwise manner (after clipping), and then shift right by 9.
                // Instead, we shift left by 7, and use mulhi, stripping the bottom 16 bits, effectively shifting right by 16,
                // resulting in a net shift of 9 bits. We use mulhi because it maintains the sign of the multiplication (unlike mullo),
                // allowing us to make use of packus to clip 2 of the inputs, resulting in a save of 2 "vec_max_16" calls.
                // A special case is when we use NEON, where we shift left by 6 instead, because the instruction "vqdmulhq_s16"
                // also doubles the return value after the multiplication, adding an extra shift to the left by 1, so we
                // compensate by shifting less before the multiplication.
                const invec_t sum0a = vec_max_16(vec_min_16(in0[i * 2 + 0], One), Zero);
                const invec_t sum1a = vec_min_16(in1[i * 2 + 0], One);
                const invec_t sum0b = vec_max_16(vec_min_16(in0[i * 2 + 1], One), Zero);
                const invec_t sum1b = vec_min_16(in1[i * 2 + 1], One);

                #if defined(USE_NEON)
                constexpr int shift = 6;
                #else
                constexpr int shift = 7;
                #endif

                const invec_t pa = vec_mulhi_16(vec_slli_16(sum0a, shift), sum1a);
                const invec_t pb = vec_mulhi_16(vec_slli_16(sum0b, shift), sum1b);

                const u32vec_t activated = reinterpret_cast<u32vec_t>(vec_packus_16(pa, pb));
                const uint16_t mask = vec_nnz(activated);
                const auto nnzs = lookup_indices[mask];
                const auto nnz_count = popcount(mask);
                for (IndexType j = 0; j < nnz_count; ++j)
                {
                    const auto    index = nnzs[j];
                    const u8vec_t in = vec_broadcast_nth_u32(activated, index);
                    const auto    col =
                      reinterpret_cast<const u8vec_t*>(&weights[(index * ChunkSize + offset + i * sizeof(u8vec_t)) * OutputDimensions]);
                    for (IndexType k = 0; k < NumRegs; ++k)
                        vec_add_dpbusd_32(sum[k], in, col[k]);
                }
            }

            offset += InputDimensions / 2;
        }

        outvec_t* outptr = reinterpret_cast<outvec_t*>(output);
        for (IndexType k = 0; k < NumRegs; ++k)
            outptr[k] = sum[k];

    #undef vec_mulhi_16
    #undef vec_zero
    #undef vec_set_16
    #undef vec_max_16
    #undef vec_min_16
    #undef vec_slli_16
    #undef vec_packus_16
    #undef vec_broadcastd_32
    #undef vec_add_dpbusd_32
    #undef vec_nnz
#else
        IndexType offset = 0;

    #if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
        AffineType
          transformedFeaturesUnaligned[FeatureTransformer<FTDimensions, nullptr>::BufferSize
                                       + alignment / sizeof(AffineType)];

        auto* transformedFeatures = align_ptr_up<alignment>(&transformedFeaturesUnaligned[0]);
    #else
        alignas(alignment) AffineType
          transformedFeatures[FeatureTransformer<FTDimensions, nullptr>::BufferSize];
    #endif

        ASSERT_ALIGNED(transformedFeatures, alignment);

        for (InputType* acc : {us, them}) {
            for (IndexType j = 0; j < InputDimensions / 2; ++j) {
                BiasType sum0 = acc[j + 0];
                BiasType sum1 = acc[j + InputDimensions / 2];
                sum0               = std::clamp<BiasType>(sum0, 0, 127 * 2);
                sum1               = std::clamp<BiasType>(sum1, 0, 127 * 2);
                transformedFeatures[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
            }
            offset += InputDimensions / 2;
        }

        // Use dense implementation for the other architectures.
        affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions, OutputDimensions>(output, weights, biases, transformedFeatures);
#endif
    }

   private:
    using BiasType   = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
