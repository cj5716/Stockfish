/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2023 The Stockfish developers (see AUTHORS file)

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


#include "psqt.h"

#include <algorithm>

#include "bitboard.h"
#include "types.h"

namespace Stockfish {

int k1 = 234, k2 = 1, k3 = 338, k4 = 51, k5 = 312, k6 = 86, k7 = 186, k8 = 84, k9 = 195, k10 = 69, k11 = 264, k12 = 83, k13 = 294, k14 = 35, k15 = 269, k16 = 1, k17 = 262, k18 = 53, k19 = 294, k20 = 90, k21 = 236, k22 = 132, k23 = 177, k24 = 124, k25 = 172, k26 = 131, k27 = 240, k28 = 130, k29 = 291, k30 = 102, k31 = 300, k32 = 59, k33 = 198, k34 = 82, k35 = 245, k36 = 144, k37 = 169, k38 = 188, k39 = 122, k40 = 188, k41 = 112, k42 = 191, k43 = 157, k44 = 161, k45 = 256, k46 = 115, k47 = 187, k48 = 83, k49 = 154, k50 = 106, k51 = 206, k52 = 145, k53 = 108, k54 = 163, k55 = 89, k56 = 170, k57 = 95, k58 = 186, k59 = 132, k60 = 184, k61 = 226, k62 = 165, k63 = 170, k64 = 113, k65 = 174, k66 = 90, k67 = 179, k68 = 170, k69 = 92, k70 = 207, k71 = 67, k72 = 210, k73 = 63, k74 = 223, k75 = 91, k76 = 180, k77 = 175, k78 = 176, k79 = 144, k80 = 93, k81 = 141, k82 = 86, k83 = 151, k84 = 158, k85 = 85, k86 = 183, k87 = 29, k88 = 192, k89 = 25, k90 = 191, k91 = 71, k92 = 201, k93 = 137, k94 = 182, k95 = 109, k96 = 103, k97 = 79, k98 = 41, k99 = 127, k100 = 122, k101 = 58, k102 = 124, k103 = 33, k104 = 131, k105 = 32, k106 = 106, k107 = 57, k108 = 116, k109 = 127, k110 = 123, k111 = 89, k112 = 47, k113 = 63, k114 = 9, k115 = 93, k116 = 56, k117 = 43, k118 = 73, k119 = -1, k120 = 78, k121 = -1, k122 = 74, k123 = 47, k124 = 67, k125 = 86, k126 = 55, k127 = 61, k128 = 9;

TUNE(k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31, k32, k33, k34, k35, k36, k37, k38, k39, k40, k41, k42, k43, k44, k45, k46, k47, k48, k49, k50, k51, k52, k53, k54, k55, k56, k57, k58, k59, k60, k61, k62, k63, k64, k65, k66, k67, k68, k69, k70, k71, k72, k73, k74, k75, k76, k77, k78, k79, k80, k81, k82, k83, k84, k85, k86, k87, k88, k89, k90, k91, k92, k93, k94, k95, k96, k97, k98, k99, k100, k101, k102, k103, k104, k105, k106, k107, k108, k109, k110, k111, k112, k113, k114, k115, k116, k117, k118, k119, k120, k121, k122, k123, k124, k125, k126, k127, k128);

namespace
{

auto constexpr S = make_score;

// 'Bonus' contains Piece-Square parameters.
// Scores are explicit for files A to D, implicitly mirrored for E to H.
constexpr Score Bonus[][RANK_NB][int(FILE_NB) / 2] = {
  { },
  { },
  { // Knight
   { S(-175, -96), S(-92,-65), S(-74,-49), S(-73,-21) },
   { S( -77, -67), S(-41,-54), S(-27,-18), S(-15,  8) },
   { S( -61, -40), S(-17,-27), S(  6, -8), S( 12, 29) },
   { S( -35, -35), S(  8, -2), S( 40, 13), S( 49, 28) },
   { S( -34, -45), S( 13,-16), S( 44,  9), S( 51, 39) },
   { S(  -9, -51), S( 22,-44), S( 58,-16), S( 53, 17) },
   { S( -67, -69), S(-27,-50), S(  4,-51), S( 37, 12) },
   { S(-201,-100), S(-83,-88), S(-56,-56), S(-26,-17) }
  },
  { // Bishop
   { S(-37,-40), S(-4 ,-21), S( -6,-26), S(-16, -8) },
   { S(-11,-26), S(  6, -9), S( 13,-12), S(  3,  1) },
   { S(-5 ,-11), S( 15, -1), S( -4, -1), S( 12,  7) },
   { S(-4 ,-14), S(  8, -4), S( 18,  0), S( 27, 12) },
   { S(-8 ,-12), S( 20, -1), S( 15,-10), S( 22, 11) },
   { S(-11,-21), S(  4,  4), S(  1,  3), S(  8,  4) },
   { S(-12,-22), S(-10,-14), S(  4, -1), S(  0,  1) },
   { S(-34,-32), S(  1,-29), S(-10,-26), S(-16,-17) }
  },
  { // Rook
   { S(-31, -9), S(-20,-13), S(-14,-10), S(-5, -9) },
   { S(-21,-12), S(-13, -9), S( -8, -1), S( 6, -2) },
   { S(-25,  6), S(-11, -8), S( -1, -2), S( 3, -6) },
   { S(-13, -6), S( -5,  1), S( -4, -9), S(-6,  7) },
   { S(-27, -5), S(-15,  8), S( -4,  7), S( 3, -6) },
   { S(-22,  6), S( -2,  1), S(  6, -7), S(12, 10) },
   { S( -2,  4), S( 12,  5), S( 16, 20), S(18, -5) },
   { S(-17, 18), S(-19,  0), S( -1, 19), S( 9, 13) }
  },
  { // Queen
   { S( 3,-69), S(-5,-57), S(-5,-47), S( 4,-26) },
   { S(-3,-54), S( 5,-31), S( 8,-22), S(12, -4) },
   { S(-3,-39), S( 6,-18), S(13, -9), S( 7,  3) },
   { S( 4,-23), S( 5, -3), S( 9, 13), S( 8, 24) },
   { S( 0,-29), S(14, -6), S(12,  9), S( 5, 21) },
   { S(-4,-38), S(10,-18), S( 6,-11), S( 8,  1) },
   { S(-5,-50), S( 6,-27), S(10,-24), S( 8, -8) },
   { S(-2,-74), S(-2,-52), S( 1,-43), S(-2,-34) }
  },
  { }
};

constexpr Score PBonus[RANK_NB][FILE_NB] =
  { // Pawn (asymmetric distribution)
   { },
   { S(  2, -8), S(  4, -6), S( 11,  9), S( 18,  5), S( 16, 16), S( 21,  6), S(  9, -6), S( -3,-18) },
   { S( -9, -9), S(-15, -7), S( 11,-10), S( 15,  5), S( 31,  2), S( 23,  3), S(  6, -8), S(-20, -5) },
   { S( -3,  7), S(-20,  1), S(  8, -8), S( 19, -2), S( 39,-14), S( 17,-13), S(  2,-11), S( -5, -6) },
   { S( 11, 12), S( -4,  6), S(-11,  2), S(  2, -6), S( 11, -5), S(  0, -4), S(-12, 14), S(  5,  9) },
   { S(  3, 27), S(-11, 18), S( -6, 19), S( 22, 29), S( -8, 30), S( -5,  9), S(-14,  8), S(-11, 14) },
   { S( -7, -1), S(  6,-14), S( -2, 13), S(-11, 22), S(  4, 24), S(-14, 17), S( 10,  7), S( -9,  7) }
  };


Score KBonus[RANK_NB][FILE_NB] =
  { // King (asymmetric distribution)
   { S(  k1,  k2), S(  k3,  k4), S(  k5,  k6), S(  k7,  k8), S( k9,  k10), S( k11, k12), S( k13, k14), S( k15, k16) },
   { S( k17, k18), S( k19, k20), S( k21, k22), S( k23, k24), S( k25, k26), S( k27, k28), S( k29, k30), S( k31, k32) },
   { S( k33, k34), S( k35, k36), S( k37, k38), S( k39, k40), S( k41, k42), S( k43, k44), S( k45, k46), S( k47, k48) },
   { S( k49, k50), S( k51, k52), S( k53, k54), S( k55, k56), S( k57, k58), S( k59, k60), S( k61, k62), S( k63, k64) },
   { S( k65, k66), S( k67, k68), S( k69, k70), S( k71, k72), S( k73, k74), S( k75, k76), S( k77, k78), S( k79, k80) },
   { S( k81, k82), S( k83, k84), S( k85, k86), S( k87, k88), S( k89, k90), S( k91, k92), S( k93, k94), S( k95, k96) },
   { S( k97, k98), S( k99,k100), S(k101,k102), S(k103,k104), S(k105,k106), S(k107,k108), S(k109,k110), S(k111,k112) },
   { S(k113,k114), S(k115,k116), S(k117,k118), S(k119,k120), S(k121,k122), S(k123,k124), S(k125,k126), S(k127,k128) }
  };

} // namespace


namespace PSQT
{

Score psq[PIECE_NB][SQUARE_NB];

// PSQT::init() initializes piece-square tables: the white halves of the tables are
// copied from Bonus[] and PBonus[], adding the piece value, then the black halves of
// the tables are initialized by flipping and changing the sign of the white scores.
void init() {

  for (Piece pc : {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING})
  {
    Score score = make_score(PieceValue[MG][pc], PieceValue[EG][pc]);

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
      File f = File(edge_distance(file_of(s)));
      psq[ pc][s] = score + ( type_of(pc) == PAWN ? PBonus[rank_of(s)][file_of(s)]
                            : type_of(pc) == KING ? KBonus[rank_of(s)][file_of(s)]
                                                  : Bonus[pc][rank_of(s)][f]);
      psq[~pc][flip_rank(s)] = -psq[pc][s];
    }
  }
}

} // namespace PSQT

} // namespace Stockfish
