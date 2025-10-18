/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

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

//Definition of input features FullThreats of NNUE evaluation function

#include "full_threats.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"

namespace Stockfish::Eval::NNUE::Features {

// Lookup array for indexing threats
IndexType offsets[PIECE_NB][SQUARE_NB + 2];

void init_threat_offsets() {
    int       cumulativeOffset     = 0;
    PieceType idxToPiece[PIECE_NB] = {
      NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE,
      NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE};

    for (int pieceIdx = 0; pieceIdx < 16; pieceIdx++)
    {
        if (idxToPiece[pieceIdx] == NO_PIECE_TYPE)
            continue;

        int cumulativePieceOffset = 0;

        for (Square from = SQ_A1; from <= SQ_H8; ++from)
        {
            offsets[pieceIdx][from] = cumulativePieceOffset;

            if (idxToPiece[pieceIdx] != PAWN)
            {
                Bitboard attacks = attacks_bb(idxToPiece[pieceIdx], from, 0ULL);
                cumulativePieceOffset += popcount(attacks);
            }

            else if (from >= SQ_A2 && from <= SQ_H7)
            {
                Bitboard attacks = (pieceIdx < 8) ? pawn_attacks_bb<WHITE>(square_bb(from))
                                                  : pawn_attacks_bb<BLACK>(square_bb(from));
                cumulativePieceOffset += popcount(attacks);
            }
        }

        offsets[pieceIdx][64] = cumulativePieceOffset;
        offsets[pieceIdx][65] = cumulativeOffset;

        cumulativeOffset += numValidTargets[pieceIdx] * cumulativePieceOffset;
    }
}

// Index of a feature for a given king position and another piece on some square
template<Color Perspective>
IndexType FullThreats::make_index(Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
    bool enemy = (attkr ^ attkd) == 8;
    from       = (Square) (int(from) ^ OrientTBL[Perspective][ksq]);
    to         = (Square) (int(to) ^ OrientTBL[Perspective][ksq]);

    if (Perspective == BLACK)
    {
        attkr = ~attkr;
        attkd = ~attkd;
    }

    // Some threats imply the existence of the corresponding ones in the opposite
    // direction. We filter them here to ensure only one such threat is active.
    if ((map[type_of(attkr) - 1][type_of(attkd) - 1] < 0)
        || (type_of(attkr) == type_of(attkd) && (enemy || type_of(attkr) != PAWN) && from < to))
    {
        return Dimensions;
    }

    Bitboard attacks = attacks_bb(attkr, from);

    return IndexType(
      offsets[attkr][65]
      + (color_of(attkd) * (numValidTargets[attkr] / 2) + map[type_of(attkr) - 1][type_of(attkd) - 1])
          * offsets[attkr][64]
      + offsets[attkr][from] + popcount((square_bb(to) - 1) & attacks));
}

// Get a list of indices for active features in ascending order
template<Color Perspective, Color C>
void FullThreats::append_active_pawn_indices(const Position &pos, IndexList& active, Square ksq) {
    constexpr Color Side     = Color(Perspective ^ C);
    constexpr Piece PawnType = make_piece(Side, PAWN);
    constexpr auto Right     = (Side == WHITE) ? NORTH_EAST : SOUTH_WEST;
    constexpr auto Left      = (Side == WHITE) ? NORTH_WEST : SOUTH_EAST;

    Bitboard bb = pos.pieces(Side, PAWN);
    Bitboard occupied = pos.pieces();
    Bitboard attacks_right = shift<Right>(bb) & occupied;
    Bitboard attacks_left  = shift<Left >(bb) & occupied;

    while (attacks_right)
    {
        Square to   = pop_lsb(attacks_right);
        Square from = to - Right;
        Piece attkd = pos.piece_on(to);
        IndexType index = make_index<Perspective>(PawnType, from, to, attkd, ksq);
        if (index < Dimensions)
            active.push_back(index);
    }

    while (attacks_left)
    {
        Square to   = pop_lsb(attacks_left);
        Square from = to - Left;
        Piece attkd = pos.piece_on(to);
        IndexType index = make_index<Perspective>(PawnType, from, to, attkd, ksq);
        if (index < Dimensions)
            active.push_back(index);
    }
}

template<Color Perspective>
void FullThreats::append_active_indices(const Position& pos, IndexList& active) {
    Square   ksq         = pos.square<KING>(Perspective);
    Bitboard occupied    = pos.pieces();

    append_active_pawn_indices<Perspective, WHITE>(pos, active, ksq);
    append_active_pawn_indices<Perspective, BLACK>(pos, active, ksq);

    // Append non-pawn indices
    Bitboard bb = occupied ^ pos.pieces(PAWN);
    while (bb)
    {
        Square from = pop_lsb(bb);
        Piece attkr = pos.piece_on(from);
        Bitboard attacks = pos.attacks_by_sq(from) & occupied;

        while (attacks)
        {
            Square to = pop_lsb(attacks);
            Piece attkd = pos.piece_on(to);
            IndexType index = make_index<Perspective>(attkr, from, to, attkd, ksq);
            if (index < Dimensions)
                active.push_back(index);
        }
    }
}

// Explicit template instantiations
template void FullThreats::append_active_indices<WHITE>(const Position& pos, IndexList& active);
template void FullThreats::append_active_indices<BLACK>(const Position& pos, IndexList& active);
template IndexType
FullThreats::make_index<WHITE>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);
template IndexType
FullThreats::make_index<BLACK>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);

// Get a list of indices for recently changed features
template<Color Perspective>
void FullThreats::append_changed_indices(Square          ksq,
                                         const DiffType& diff,
                                         IndexList&      removed,
                                         IndexList&      added) {
    for (const auto [attacker, attacked, from, to, add] : diff.list)
    {
        IndexType index = make_index<Perspective>(attacker, from, to, attacked, ksq);

        if (index == Dimensions)
            continue;

        if (add)
            added.push_back(index);
        else
            removed.push_back(index);
    }
}

// Explicit template instantiations
template void FullThreats::append_changed_indices<WHITE>(Square          ksq,
                                                         const DiffType& diff,
                                                         IndexList&      removed,
                                                         IndexList&      added);
template void FullThreats::append_changed_indices<BLACK>(Square          ksq,
                                                         const DiffType& diff,
                                                         IndexList&      removed,
                                                         IndexList&      added);

bool FullThreats::requires_refresh(const DiffType& diff, Color perspective) {
    return perspective == diff.us
        && OrientTBL[diff.us][diff.ksq] != OrientTBL[diff.us][diff.prevKsq];
}

}  // namespace Stockfish::Eval::NNUE::Features
