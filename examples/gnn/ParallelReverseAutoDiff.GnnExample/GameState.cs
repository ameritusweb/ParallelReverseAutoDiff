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
        private ConcurrentDictionary<(Position, char), List<(Position, MoveType, char?, char?)>> positionToPossibleMoveMap;
        private ConcurrentDictionary<Position, GNNNode> positionToNodeMap;
        private GNNGraph graph;

        /// <summary>
        /// Initializes a new instance of the <see cref="GameState"/> class.
        /// </summary>
        public GameState()
        {
            this.Board = new ChessBoard();
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Position, MoveType, char?, char?)>>();
            this.positionToNodeMap = new ConcurrentDictionary<Position, GNNNode>();
            this.graph = new GNNGraph();
            this.BuildMap();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GameState"/> class.
        /// </summary>
        /// <param name="board">The chess board.</param>
        public GameState(ChessBoard board)
        {
            this.Board = board;
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Position, MoveType, char?, char?)>>();
            this.graph = new GNNGraph();
            this.BuildMap();
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

        /// <summary>
        /// Gets all positions on the chess board.
        /// </summary>
        /// <returns>The list of positions.</returns>
        public List<Position> GetAllPositions()
        {
            var positions = new List<Position>();
            for (var i = 0; i < 64; i++)
            {
                positions.Add(new Position((short)(i / 8), (short)(i % 8)));
            }

            return positions;
        }

        /// <summary>
        /// Gets all king positions from a position.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <returns>The list of positions.</returns>
        public List<Position> GetAllKingPositionsFrom(Position position)
        {
            var positions = new List<Position>();
            for (var i = -1; i <= 1; ++i)
            {
                for (var j = -1; j <= 1; ++j)
                {
                    if (i == 0 && j == 0)
                    {
                        continue;
                    }

                    var pos = new Position((short)(position.RankValue + i), (short)(position.FileValue + j));
                    if (this.Board.IsValid(pos))
                    {
                        positions.Add(pos);
                    }
                }
            }

            if (position.RankValue == 0 || position.RankValue == 7)
            {
                if (position.FileValue == 3 || position.FileValue == 4)
                {
                    var castle1 = new Position(position.RankValue, (short)(position.FileValue + 2));
                    if (this.Board.IsValid(castle1))
                    {
                        positions.Add(castle1);
                    }

                    var castle2 = new Position(position.RankValue, (short)(position.FileValue - 2));
                    if (this.Board.IsValid(castle2))
                    {
                        positions.Add(castle2);
                    }
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets all rook positions from a position.
        /// </summary>
        /// <param name="position">A position.</param>
        /// <returns>The rook positions.</returns>
        public List<Position> GetAllRookPositionsFrom(Position position)
        {
            var positions = new List<Position>();
            for (var i = -7; i <= 7; ++i)
            {
                if (i == 0)
                {
                    continue;
                }

                var pos = new Position((short)(position.RankValue + i), position.FileValue);
                if (this.Board.IsValid(pos))
                {
                    positions.Add(pos);
                }
            }

            for (int j = -7; j <= 7; ++j)
            {
                if (j == 0)
                {
                    continue;
                }

                var pos = new Position(position.RankValue, (short)(position.FileValue + j));
                if (this.Board.IsValid(pos))
                {
                    positions.Add(pos);
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets all bishop positions from a position.
        /// </summary>
        /// <param name="position">A position.</param>
        /// <returns>The bishop positions.</returns>
        public List<Position> GetAllBishopPositionsFrom(Position position)
        {
            var positions = new List<Position>();

            for (var i = -7; i <= 7; ++i)
            {
                for (var j = -7; j <= 7; ++j)
                {
                    if (i == 0 && j == 0)
                    {
                        continue;
                    }

                    var pos = new Position((short)(position.RankValue + i), (short)(position.FileValue + j));
                    if (this.Board.IsValid(pos))
                    {
                        positions.Add(pos);
                    }
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets all queen positions from a position.
        /// </summary>
        /// <param name="position">A position.</param>
        /// <returns>The queen positions.</returns>
        public List<Position> GetAllQueenPositionsFrom(Position position)
        {
            var positions = new List<Position>();
            for (var i = -7; i <= 7; ++i)
            {
                if (i == 0)
                {
                    continue;
                }

                var pos = new Position((short)(position.RankValue + i), position.FileValue);
                if (this.Board.IsValid(pos))
                {
                    positions.Add(pos);
                }
            }

            for (int j = -7; j <= 7; ++j)
            {
                if (j == 0)
                {
                    continue;
                }

                var pos = new Position(position.RankValue, (short)(position.FileValue + j));
                if (this.Board.IsValid(pos))
                {
                    positions.Add(pos);
                }
            }

            for (var i = -7; i <= 7; ++i)
            {
                for (var j = -7; j <= 7; ++j)
                {
                    if (i == 0 && j == 0)
                    {
                        continue;
                    }

                    var pos = new Position((short)(position.RankValue + i), (short)(position.FileValue + j));
                    if (this.Board.IsValid(pos))
                    {
                        positions.Add(pos);
                    }
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets all of the pawn positions from a position.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <returns>The list of positions.</returns>
        public List<Position> GetAllPawnPositionsFrom(Position position)
        {
            var positions = new List<Position>();
            for (var i = -1; i <= 1; ++i)
            {
                for (var j = -1; j <= 1; ++j)
                {
                    if (i == 0 && j == 0)
                    {
                        continue;
                    }

                    var pos = new Position((short)(position.RankValue + i), (short)(position.FileValue + j));
                    if (this.Board.IsValid(pos))
                    {
                        positions.Add(pos);
                    }
                }
            }

            for (var i = -2; i <= 2; ++i)
            {
                if (i == 0)
                {
                    continue;
                }

                var pos = new Position((short)(position.RankValue + i), position.FileValue);
                if (this.Board.IsValid(pos))
                {
                    positions.Add(pos);
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets all knight positions from a position.
        /// </summary>
        /// <param name="position">A position.</param>
        /// <returns>The knight positions.</returns>
        public List<Position> GetAllKnightPositionsFrom(Position position)
        {
            var positions = new List<Position>();
            for (var i = -2; i <= 2; ++i)
            {
                if (i == 0)
                {
                    continue;
                }

                for (var j = -2; j <= 2; ++j)
                {
                    if (j == 0)
                    {
                        continue;
                    }

                    if (Math.Abs(i) == Math.Abs(j))
                    {
                        continue;
                    }

                    var pos = new Position((short)(position.RankValue + i), (short)(position.FileValue + j));
                    if (this.Board.IsValid(pos))
                    {
                        positions.Add(pos);
                    }
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets all piece types.
        /// </summary>
        /// <returns>A list of piece types.</returns>
        public List<PieceType> GetAllPieceTypes()
        {
            return new List<PieceType>()
            {
                PieceType.Pawn,
                PieceType.Knight,
                PieceType.Bishop,
                PieceType.Rook,
                PieceType.Queen,
                PieceType.King,
            };
        }

        /// <summary>
        /// Add position to map.
        /// </summary>
        /// <param name="startPos">The start position.</param>
        /// <param name="moveInfo">The move info.</param>
        public void AddToMap((Position Pos, char PieceType) startPos, (Position NewPos, MoveType MoveType, char? CapturePieceType, char? PromotionPieceType) moveInfo)
        {
            if (this.positionToPossibleMoveMap.ContainsKey(startPos))
            {
                var list = this.positionToPossibleMoveMap[startPos];
                list.Add(moveInfo);
                this.positionToPossibleMoveMap.AddOrUpdate(startPos, list, (key, oldValue) => list);
            }
            else
            {
                this.positionToPossibleMoveMap.TryAdd(startPos, new List<(Position NewPos, MoveType MoveType, char? CapturePieceType, char? PromotionPieceType)> { moveInfo });
            }

            var nodeFrom = this.positionToNodeMap[startPos.Pos];
            var nodeTo = this.positionToNodeMap[moveInfo.NewPos];
            var edge = new GNNEdge(nodeFrom, nodeTo);

            this.graph.Edges.Add(edge);
            nodeFrom.Edges.Add(edge);
            nodeTo.Edges.Add(edge);
        }

        private void BuildMap()
        {
            var positions = this.GetAllPositions();
            var pieceTypes = this.GetAllPieceTypes();
            for (int i = 0; i < positions.Count; ++i)
            {
                var position = positions[i];
                GNNNode node = new GNNNode(position);
                this.positionToNodeMap.AddOrUpdate(position, node, (key, oldValue) => node);
                this.graph.Nodes.Add(node);
            }

            for (int i = 0; i < positions.Count; ++i)
            {
                var position = positions[i];
                for (int j = 0; j < pieceTypes.Count; ++j)
                {
                    var pieceType = pieceTypes[j];
                    var key = (position, pieceType.AsChar);
                    switch (pieceType.AsChar)
                    {
                        case 'p':
                            {
                                var pawnPositions = this.GetAllPawnPositionsFrom(position);
                                for (int k = 0; k < pawnPositions.Count; ++k)
                                {
                                    var pawnPosition = pawnPositions[k];
                                    this.AddToMap(key, (pawnPosition, MoveType.Defense, null, null));
                                    this.AddToMap(key, (pawnPosition, MoveType.QueensideCastle, null, null));
                                    this.AddToMap(key, (pawnPosition, MoveType.KingsideCastle, null, null));
                                    this.AddToMap(key, (pawnPosition, MoveType.EnPassant, PieceType.Pawn.AsChar, null));
                                    this.AddToMap(key, (pawnPosition, MoveType.None, null, null));
                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (pawnPosition, MoveType.Capture, pieceTypes[l].AsChar, null));
                                        for (int m = 0; m < pieceTypes.Count; ++m)
                                        {
                                            this.AddToMap(key, (pawnPosition, MoveType.Capture, pieceTypes[l].AsChar, pieceTypes[m].AsChar));
                                        }
                                    }

                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (pawnPosition, MoveType.Promotion, null, pieceTypes[l].AsChar));
                                    }
                                }

                                break;
                            }

                        case 'n':
                            {
                                var knightPositions = this.GetAllKnightPositionsFrom(position);
                                for (int k = 0; k < knightPositions.Count; ++k)
                                {
                                    var knightPosition = knightPositions[k];
                                    this.AddToMap(key, (knightPosition, MoveType.Defense, null, null));
                                    this.AddToMap(key, (knightPosition, MoveType.None, null, null));
                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (knightPosition, MoveType.Capture, pieceTypes[l].AsChar, null));
                                    }
                                }

                                break;
                            }

                        case 'b':
                            {
                                var bishopPositions = this.GetAllBishopPositionsFrom(position);
                                for (int k = 0; k < bishopPositions.Count; ++k)
                                {
                                    var bishopPosition = bishopPositions[k];
                                    this.AddToMap(key, (bishopPosition, MoveType.Defense, null, null));
                                    this.AddToMap(key, (bishopPosition, MoveType.None, null, null));
                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (bishopPosition, MoveType.Capture, pieceTypes[l].AsChar, null));
                                    }
                                }

                                break;
                            }

                        case 'r':
                            {
                                var rookPositions = this.GetAllRookPositionsFrom(position);
                                for (int k = 0; k < rookPositions.Count; ++k)
                                {
                                    var rookPosition = rookPositions[k];
                                    this.AddToMap(key, (rookPosition, MoveType.Defense, null, null));
                                    this.AddToMap(key, (rookPosition, MoveType.None, null, null));
                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (rookPosition, MoveType.Capture, pieceTypes[l].AsChar, null));
                                    }
                                }

                                break;
                            }

                        case 'q':
                            {
                                var queenPositions = this.GetAllQueenPositionsFrom(position);
                                for (int k = 0; k < queenPositions.Count; ++k)
                                {
                                    var queenPosition = queenPositions[k];
                                    this.AddToMap(key, (queenPosition, MoveType.Defense, null, null));
                                    this.AddToMap(key, (queenPosition, MoveType.None, null, null));
                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (queenPosition, MoveType.Capture, pieceTypes[l].AsChar, null));
                                    }
                                }

                                break;
                            }

                        case 'k':
                            {
                                var kingPositions = this.GetAllKingPositionsFrom(position);
                                for (int k = 0; k < kingPositions.Count; ++k)
                                {
                                    var kingPosition = kingPositions[k];
                                    this.AddToMap(key, (kingPosition, MoveType.Defense, null, null));
                                    this.AddToMap(key, (kingPosition, MoveType.None, null, null));
                                    this.AddToMap(key, (kingPosition, MoveType.KingsideCastle, null, null));
                                    this.AddToMap(key, (kingPosition, MoveType.QueensideCastle, null, null));
                                    for (int l = 0; l < pieceTypes.Count; ++l)
                                    {
                                        this.AddToMap(key, (kingPosition, MoveType.Capture, pieceTypes[l].AsChar, null));
                                    }
                                }

                                break;
                            }
                    }
                }
            }
        }
    }
}
