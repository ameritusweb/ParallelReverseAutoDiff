//------------------------------------------------------------------------------
// <copyright file="GameState.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace RandomForestTest
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using Chess;
    using ParallelReverseAutoDiff.GnnExample.GNN;

    /// <summary>
    /// The chess game state.
    /// </summary>
    public class GameState
    {
        private Random rng;
        private ConcurrentDictionary<(Position, char), List<(Move, char)>> positionToPossibleMoveMap;
        private ConcurrentDictionary<Position, GNNNode> positionToPieceMap;
        private ConcurrentDictionary<Position, GNNNode> positionToSquareMap;
        private GNNGraph graph;

        /// <summary>
        /// Initializes a new instance of the <see cref="GameState"/> class.
        /// </summary>
        public GameState()
        {
            this.Board = new ChessBoard();
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Move, char)>>();
            this.positionToPieceMap = new ConcurrentDictionary<Position, GNNNode>();
            this.positionToSquareMap = new ConcurrentDictionary<Position, GNNNode>();
            this.graph = new GNNGraph();
            this.BuildGraph();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GameState"/> class.
        /// </summary>
        /// <param name="board">The chess board.</param>
        public GameState(ChessBoard board)
        {
            this.Board = board;
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Move, char)>>();
            this.positionToPieceMap = new ConcurrentDictionary<Position, GNNNode>();
            this.positionToSquareMap = new ConcurrentDictionary<Position, GNNNode>();
            this.graph = new GNNGraph();
            this.BuildGraph();
        }

        /// <summary>
        /// Gets or sets the chess game board.
        /// </summary>
        public ChessBoard Board { get; set; }

        /// <summary>
        /// Gets or sets the board size.
        /// </summary>
        public int BoardSize { get; set; } = 8;

        /// <summary>
        /// Gets the king square.
        /// </summary>
        /// <param name="color">The piece color.</param>
        /// <returns>The position.</returns>
        public Position? GetKingSquare(PieceColor color)
        {
            return color == PieceColor.White ? this.Board.WhiteKing : this.Board.BlackKing;
        }

        /// <summary>
        /// Gets the piece at the position.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <returns>The piece.</returns>
        public Piece? GetPieceAt(Position position)
        {
            return this.Board[position];
        }

        /// <summary>
        /// Gets the piece at the position.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <returns>The piece.</returns>
        public Piece? GetPieceAt(int position)
        {
            return this.Board[new Position((short)(position / 8), (short)(position % 8))];
        }

        /// <summary>
        /// Get all possible moves.
        /// </summary>
        /// <returns>The list of moves.</returns>
        public List<Move> GetAllMoves()
        {
            return this.Board.Moves(false, true, false).ToList();
        }

        /// <summary>
        /// Get all moves for a piece color.
        /// </summary>
        /// <param name="color">The color.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> GetAllMovesForColor(PieceColor color)
        {
            return this.Board.Moves(false, true, false)
                .Where(x => x.Piece.Color == color)
                .ToList();
        }

        /// <summary>
        /// Get all moves for a piece's color and type.
        /// </summary>
        /// <param name="color">The color.</param>
        /// <param name="type">The type.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> GetAllMovesForColorAndType(PieceColor color, PieceType type)
        {
            return this.Board.Moves(false, true, false)
                .Where(x => x.Piece.Color == color && x.Piece.Type == type)
                .ToList();
        }

        /// <summary>
        /// Get all of the legal moves for a position integer.
        /// </summary>
        /// <param name="position">The position integer.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> GetLegalMoves(int position)
        {
            var pos = new Position((short)(position / 8), (short)(position % 8));
            return this.Board.Moves().Where(x => x.OriginalPosition == pos).ToList();
        }

        /// <summary>
        /// Get all the legal moves for a certain position.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> GetLegalMoves(Position position)
        {
            return this.Board.Moves().Where(x => x.OriginalPosition == position).ToList();
        }

        /// <summary>
        /// Gets the legal moves for a position and color.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="color">The color.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> GetLegalMovesForPositionAndColor(Position position, PieceColor color)
        {
            return this.Board.Moves().Where(x => x.OriginalPosition == position && x.Piece.Color == color).ToList();
        }

        /// <summary>
        /// Gets all possible moves for the position and color.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="color">The color.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> GetAllMovesForPositionAndColor(Position position, PieceColor color)
        {
            return this.Board.Moves(false, true, false).Where(x => x.OriginalPosition == position && x.Piece.Color == color).ToList();
        }

        /// <summary>
        /// Makes a random and move and returns it.
        /// </summary>
        /// <returns>The move made.</returns>
        public Move MoveRandomly()
        {
            var moves = this.Board.Moves();
            var move = this.rng.Next() % moves.Length;
            this.Board.Move(moves[move]);
            return moves[move];
        }

        /// <summary>
        /// Chooses a ramdom move from the list.
        /// </summary>
        /// <param name="moves">The list of moves.</param>
        /// <returns>The move.</returns>
        public Move MoveRandomly(List<Move> moves)
        {
            var move = this.rng.Next() % moves.Count;
            this.Board.Move(moves[move]);
            return moves[move];
        }

        /// <summary>
        /// Get the last number of moves.
        /// </summary>
        /// <param name="numOfMoves">The number of moves.</param>
        /// <returns>The list of moves.</returns>
        public List<Move> LastMoves(int numOfMoves)
        {
            return this.Board.ExecutedMoves.TakeLast(numOfMoves).ToList();
        }

        /// <summary>
        /// Is the game over.
        /// </summary>
        /// <returns>Whether the game is over.</returns>
        public bool IsGameOver()
        {
            return this.Board.EndGame?.EndgameType == EndgameType.DrawDeclared || this.Board.EndGame?.EndgameType == EndgameType.Checkmate || this.Board.EndGame?.EndgameType == EndgameType.Stalemate;
        }

        /// <summary>
        /// Is the position valid by rank and file.
        /// </summary>
        /// <param name="rank">The rank.</param>
        /// <param name="file">The file.</param>
        /// <returns>Whether the position is valid.</returns>
        public bool IsValid(int rank, int file)
        {
            return this.Board.IsValid(new Position(rank, file));
        }

        /// <summary>
        /// Is the chess move valid.
        /// </summary>
        /// <param name="move">The move.</param>
        /// <returns>Whether the move is valid.</returns>
        public bool IsValidMove(Move move)
        {
            return this.Board.Moves().Any(x => x.ToString() == move.ToString());
        }

        private void BuildGraph()
        {
            var piecesAndPositionsAll = this.Board.GetPiecesAndTheirPositionsAll();
            foreach ((Piece? piece, Position position) pieceAndPosition in piecesAndPositionsAll)
            {
                try
                {
                    if (pieceAndPosition.piece != null)
                    {
                        var piece = pieceAndPosition.piece;
                        var position = pieceAndPosition.position;
                        ChessPieceGNNNode node = new ChessPieceGNNNode()
                        {
                            PieceColor = piece.Color,
                            PieceType = piece.Type,
                        };
                        this.positionToPieceMap.TryAdd(position, node);
                        ChessSquareGNNNode square = new ChessSquareGNNNode()
                        {
                            Position = pieceAndPosition.position,
                        };
                        this.positionToSquareMap.TryAdd(pieceAndPosition.position, square);
                    }
                    else
                    {
                        ChessSquareGNNNode node = new ChessSquareGNNNode()
                        {
                            Position = pieceAndPosition.position,
                        };
                        this.positionToSquareMap.TryAdd(pieceAndPosition.position, node);
                    }
                }
                catch (Exception) { }
            }

            foreach ((Piece? piece, Position position) pieceAndPosition in piecesAndPositionsAll)
            {
                try
                {
                    if (pieceAndPosition.piece != null)
                    {
                        var piece = pieceAndPosition.piece;
                        var position = pieceAndPosition.position;
                        ChessPieceGNNNode node = (ChessPieceGNNNode)this.positionToPieceMap[position];
                        var moveAndPieceList = pieceAndPosition.piece.GetAvailableMovesToAnyColor(this.Board, pieceAndPosition.position);
                        this.positionToPossibleMoveMap.TryAdd((position, piece.Type.AsChar), moveAndPieceList.Select(x => (x.move, x.piece.Type.AsChar)).ToList());
                        foreach (var moveAndPiece in moveAndPieceList)
                        {
                            var p = moveAndPiece.piece;
                            var m = moveAndPiece.move;
                            var newPosition = m.NewPosition;
                            var newPositionNode = this.positionToPieceMap[newPosition];
                            GNNEdge edge = new GNNEdge()
                            {
                                From = node,
                                To = newPositionNode,
                            };
                            node.Edges.Add(edge);
                            newPositionNode.Edges.Add(edge);
                        }
                    }
                }
                catch (Exception) { }
            }
        }
    }
}
