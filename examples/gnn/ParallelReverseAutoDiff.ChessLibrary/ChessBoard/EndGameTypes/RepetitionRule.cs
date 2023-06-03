// *****************************************************
// *                                                   *
// * O Lord, Thank you for your goodness in our lives. *
// *     Please bless this code to our compilers.      *
// *                     Amen.                         *
// *                                                   *
// *****************************************************
//                                    Made by Geras1mleo

using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace Chess
{
    /// <summary>
    /// https://www.chessprogramming.org/Repetitions
    /// </summary>
    internal class RepetitionRule : EndGameRule
    {
        private const int MINIMUM_MOVES_COUNT = 8; // at least 8 moves required to get threefold repetition

        public RepetitionRule(ChessBoard board) : base(board) { }

        internal override EndgameType Type => EndgameType.Repetition;

        internal override bool IsEndGame()
        {
            bool isRepetition = false;
            var movesCount = board.MoveIndex + 1;

            if (movesCount >= MINIMUM_MOVES_COUNT)
            {
                var fen = board.ToPositionFen();
                if (board.fenCounts.ContainsKey(fen) && board.fenCounts[fen] >= 3)
                {
                    isRepetition = true;
                }
            }

            return isRepetition;
        }
    }
}