// *****************************************************
// *                                                   *
// * O Lord, Thank you for your goodness in our lives. *
// *     Please bless this code to our compilers.      *
// *                     Amen.                         *
// *                                                   *
// *****************************************************
//                                    Made by Geras1mleo

using System;
using System.Collections.Generic;

namespace Chess
{
    /// <summary>
    /// Chess Piece
    /// </summary>
    public class Piece : ICloneable
    {
        /// <summary>
        /// Color of piece White/Black
        /// </summary>
        public PieceColor Color { get; }

        /// <summary>
        /// Type of piece
        /// See: PieceType SmartEnum
        /// </summary>
        public PieceType Type { get; internal set; }

        /// <summary>
        /// Initializes new Piece by given color and type
        /// </summary>
        public Piece(PieceColor color, PieceType type)
        {
            Color = color;
            Type = type;
        }

        /// <summary>
        /// Initializes new Piece by given color and type as:<br/>
        /// "wp" => White Pawn<br/>
        /// "br" => Black Rook<br/>
        /// See: piece.ToString()<br/>
        /// </summary>
        public Piece(string piece)
        {
            if (!Regexes.RegexPiece.IsMatch(piece))
            {
                throw new ChessArgumentException(null, "Piece should match pattern: " + Regexes.PiecePattern);
            }

            Color = PieceColor.FromChar(piece[0]);
            Type = PieceType.FromChar(piece[1]);
        }

        internal Piece(Piece piece)
        {
            Color = PieceColor.FromValue(piece.Color.Value);
            Type = PieceType.FromValue(piece.Type.Value);
        }

        public int MaterialValue
        {
            get
            {
                if (Type == PieceType.Pawn)
                {
                    return 1;
                }
                if (Type == PieceType.Knight || Type == PieceType.Bishop)
                {
                    return 3;
                }
                if (Type == PieceType.Rook)
                {
                    return 5;
                }
                if (Type == PieceType.Queen)
                {
                    return 9;
                }
                return 0;
            }
        }

        /// <summary>
        /// Initializes new Piece by given color and type as FEN:<br/>
        /// 'Q' => White Queen<br/>
        /// 'q' => Black Queen<br/>
        /// See: piece.ToFenChar()<br/>
        /// </summary>
        public Piece(char fenChar)
        {
            if (!Regexes.RegexFenPiece.IsMatch(fenChar.ToString()))
            {
                throw new ChessArgumentException(null, "FEN piece character should match pattern: " + Regexes.FenPiecePattern);
            }

            Type = PieceType.FromChar(fenChar);
            Color = char.IsLower(fenChar) ? PieceColor.Black : PieceColor.White;
        }

        /// <summary>
        /// Piece as:<br/>
        /// "wp" => White Pawn<br/>
        /// "br" => Black Rook<br/>
        /// See: new Piece(string piece)<br/>
        /// </summary>
        public override string ToString() => $"{Color.AsChar}{Type.AsChar}";

        /// <summary>
        /// Piece as FEN char<br/>
        /// Uppercase => White piece<br/>
        /// Lowercase => Black piece<br/>
        /// See: new Piece(char fenChar)<br/>
        /// </summary>
        public char ToFenChar()
        {
            return Color switch
            {
                var e when e == PieceColor.White => char.ToUpper(Type.AsChar),
                var e when e == PieceColor.Black => char.ToLower(Type.AsChar),
                var _ => throw new ChessArgumentException(null, nameof(Color), nameof(Piece.ToFenChar)),
            };
        }

        public object Clone()
        {
            Piece clonedPiece = new Piece(this.Color, this.Type);
            return clonedPiece;
        }

        public List<Tuple<Move, Piece>> GetAvailableMovesToTheSameColor(ChessBoard board, Position position)
        {
            List<Tuple<Move, Piece>> moves = new List<Tuple<Move, Piece>>();
            var potential = board.GeneratePositions(position, true);
            foreach (var p in potential)
            {
                var piece = board.pieces[p.Y, p.X];
                if (Type == PieceType.Pawn && position.X == p.X)
                {
                    continue;
                }
                if (piece != null && piece.Color == Color)
                {
                    moves.Add(new Tuple<Move, Piece>(new Move(position, p), piece));
                }
            }
            return moves;
        }

        public List<Tuple<Move, Piece?>> GetSquareControl(ChessBoard board, Position position)
        {
            List<Tuple<Move, Piece?>> moves = new List<Tuple<Move, Piece?>>();
            var potential = board.GeneratePositions(position, true);
            foreach (var p in potential)
            {
                var piece = board.pieces[p.Y, p.X];
                if (Type == PieceType.Pawn && position.X == p.X)
                {
                    continue;
                }
                moves.Add(new Tuple<Move, Piece?>(new Move(position, p), piece));
            }
            return moves;
        }

        public List<(Move move, Piece? piece)> GetAvailableMovesToAnyColor(ChessBoard board, Position position)
        {
            List<(Move, Piece?)> moves = new List<(Move, Piece?)>();
            var potential = board.GeneratePositions(position, true);
            foreach (var p in potential)
            {
                var piece = board.pieces[p.Y, p.X];
                //if (Type == PieceType.Pawn && position.X == p.X)
                //{
                //    continue;
                //}
                //if (piece != null)
                //{
                    moves.Add((new Move(position, p), piece));
                //}
            }
            return moves;
        }

        public bool CanAttack(Position square, ChessBoard board, Move[] allMoves)
        {
            foreach (var move in allMoves)
            {
                if (move.Piece == this)
                {
                    if (move.NewPosition == square)
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    }
}