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


constexpr Score KBonus[RANK_NB][FILE_NB] =
  { // King (asymmetric distribution)
   { S(255,  1), S(304, 49), S(279, 83), S(184, 75), S(189, 77), S(263, 91), S(304, 41), S(280,  1) },
   { S(256, 55), S(306, 96), S(244,127), S(180,123), S(165,139), S(238,135), S(289, 98), S(296, 55) },
   { S(191, 78), S(252,130), S(173,180), S(122,178), S(115,172), S(166,157), S(261,124), S(185, 91) },
   { S(160,107), S(207,162), S(119,171), S( 95,176), S( 98,182), S(140,178), S(203,155), S(175,109) },
   { S(165, 98), S(182,170), S(105,209), S( 68,209), S( 67,208), S( 93,190), S(180,170), S(131, 96) },
   { S(133, 93), S(136,167), S( 84,187), S( 29,187), S( 31,182), S( 72,200), S(144,177), S(117, 94) },
   { S( 82, 41), S(128,121), S( 62,120), S( 36,138), S( 32,129), S( 60,116), S(113,128), S( 86, 48) },
   { S( 56, 11), S( 84, 58), S( 44, 74), S( -1, 77), S( -1, 73), S( 45, 68), S( 93, 57), S( 57, 10) }
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
