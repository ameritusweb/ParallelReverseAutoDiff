//------------------------------------------------------------------------------
// <copyright file="GameState.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using Chess;
    using ManagedCuda;
    using ParallelReverseAutoDiff.GnnExample.GNN;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths;

    /// <summary>
    /// The chess game state.
    /// </summary>
    public class GameState
    {
        private Random rng;
        private ConcurrentDictionary<(Position, char), List<(Position, MoveType, char?, char?)>> positionToPossibleMoveMap;
        private ConcurrentDictionary<Position, GNNNode> positionToNodeMap;
        private ConcurrentDictionary<int, GNNEdge> idToEdgeMap;
        private GNNGraph graph;

        /// <summary>
        /// Initializes a new instance of the <see cref="GameState"/> class.
        /// </summary>
        public GameState()
        {
            this.Board = new ChessBoard();
            this.Board.AutoEndgameRules = AutoEndgameRules.All;
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Position, MoveType, char?, char?)>>();
            this.positionToNodeMap = new ConcurrentDictionary<Position, GNNNode>();
            this.idToEdgeMap = new ConcurrentDictionary<int, GNNEdge>();
            this.graph = new GNNGraph();

            // this.BuildMap();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GameState"/> class.
        /// </summary>
        /// <param name="board">The chess board.</param>
        public GameState(ChessBoard board)
        {
            this.Board = board;
            this.Board.AutoEndgameRules = AutoEndgameRules.All;
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Position, MoveType, char?, char?)>>();
            this.idToEdgeMap = new ConcurrentDictionary<int, GNNEdge>();
            this.graph = new GNNGraph();

            // this.BuildMap();
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
        /// Gets the GAP edge from the move.
        /// </summary>
        /// <param name="graph">The graph.</param>
        /// <param name="move">The move.</param>
        /// <param name="legalMoves">The legal moves.</param>
        /// <param name="turn">The turn.</param>
        /// <returns>The gap edge.</returns>
        public static (GapEdge Edge1, GapEdge Edge2) GetGapEdge(GapGraph graph, Move move, List<Move> legalMoves, PieceColor turn)
        {
            Piece piece = move.Piece;
            Position originalPosition = move.OriginalPosition;
            Position endPosition = move.NewPosition;
            Piece? capturedPiece = null;
            if (move.CapturedPiece != null)
            {
                capturedPiece = move.CapturedPiece;
            }

            bool isLegal = legalMoves.Any(x => x.OriginalPosition.ToString() == originalPosition.ToString()
                           &&
                           x.NewPosition.ToString() == endPosition.ToString()
                           &&
                           x.Piece.ToString() == piece.ToString());

            var node1 = graph.GapNodes.FirstOrDefault(n => n.PositionX == originalPosition.X && n.PositionY == originalPosition.Y) ?? throw new InvalidOperationException("Position not found.");
            var node2 = graph.GapNodes.FirstOrDefault(n => n.PositionX == endPosition.X && n.PositionY == endPosition.Y) ?? throw new InvalidOperationException("Position not found.");
            return GetEdges(node1, node2, move.ToString(), isLegal, piece.Color.ToString() == turn.ToString());
        }

        /// <summary>
        /// Gets the GAP path from the move.
        /// </summary>
        /// <param name="graph">The graph.</param>
        /// <param name="move">The move.</param>
        /// <param name="nextmove">The next move.</param>
        /// <param name="legalBothMoves">The legal both moves.</param>
        /// <param name="legalMoves">The legal moves.</param>
        /// <param name="turn">The turn.</param>
        /// <returns>The GAP path.</returns>
        public static GapPath GetGapPath(GapGraph graph, string move, Move nextmove, List<Move> legalBothMoves, List<Move> legalMoves, PieceColor turn)
        {
            Piece piece = new Piece(move.Substring(0, 2));
            Position originalPosition = new Position(move.Substring(2, 2));
            Position endPosition = new Position(move.Substring(4, 2));
            Piece? capturedPiece = null;
            if (move.Length > 6)
            {
                capturedPiece = new Piece(move.Substring(6, 2));
            }

            bool isYourTurn = false;
            if (piece.Color.ToString() == turn.ToString())
            {
                isYourTurn = true;
            }

            bool isLegal = false;
            if (legalMoves.Any(x => x.OriginalPosition.ToString() == originalPosition.ToString()
                &&
                x.NewPosition.ToString() == endPosition.ToString()
                &&
                x.Piece.ToString() == piece.ToString()))
            {
                isLegal = true;
            }

            bool isTargetMove = false;
            if (nextmove.Piece.ToString() == piece.ToString()
                &&
                nextmove.OriginalPosition.ToString() == originalPosition.ToString()
                &&
                nextmove.NewPosition.ToString() == endPosition.ToString())
            {
                if (capturedPiece == null && nextmove.CapturedPiece == null)
                {
                    isTargetMove = true;
                }
                else if (capturedPiece != null && nextmove.CapturedPiece != null && capturedPiece.ToString() == nextmove.CapturedPiece.ToString())
                {
                    isTargetMove = true;
                }
            }

            if (!legalBothMoves.Any(x => x.Piece.ToString() == piece.ToString() && x.OriginalPosition.ToString() == originalPosition.ToString() && x.NewPosition.ToString() == endPosition.ToString()))
            {
                var node1 = graph.GapNodes.FirstOrDefault(x => x.PositionX == originalPosition.X && x.PositionY == originalPosition.Y) ?? throw new InvalidOperationException("Node should not be null.");
                var node2 = graph.GapNodes.FirstOrDefault(x => x.PositionX == endPosition.X && x.PositionY == endPosition.Y) ?? throw new InvalidOperationException("Node should not be null.");
                var edges = GetEdges(node1, node2, move, false, isYourTurn);
                graph.GapEdges.Add(edges.Edge1);
                graph.GapEdges.Add(edges.Edge2);
            }

            GapPath gapPath = new GapPath()
            {
                Id = Guid.NewGuid(),
                IsTarget = isTargetMove,
                IsLegal = isLegal,
                IsYourTurn = isYourTurn,
                MoveString = move,
            };

            Move m = new Move(originalPosition, endPosition, piece, capturedPiece);
            List<Position> path = GetPath(m);
            foreach (var position in path)
            {
                var node = graph.GapNodes.FirstOrDefault(n => n.PositionX == position.FileValue && n.PositionY == position.RankValue);
                if (node != null)
                {
                    gapPath.AddNode(node);
                }
            }

            return gapPath;
        }

        /// <summary>
        /// Gets the path from a move.
        /// </summary>
        /// <param name="move">A move.</param>
        /// <returns>The list of positions.</returns>
        public static List<Position> GetPath(Move move)
        {
            List<Position> path = new List<Position>();

            // Get the piece type and positions
            PieceType pieceType = move.Piece.Type;
            Position originalPosition = move.OriginalPosition;
            Position newPosition = move.NewPosition;

            // Add the original position
            path.Add(originalPosition);

            // Check if the piece is a knight
            if (pieceType == PieceType.Knight)
            {
                // Calculate the intermediate positions
                Position intermediate1;
                Position intermediate2;

                if (Math.Abs(originalPosition.RankValue - newPosition.RankValue) == 2)
                {
                    intermediate1 = new Position(originalPosition.RankValue + Math.Sign(newPosition.RankValue - originalPosition.RankValue), originalPosition.FileValue);
                    intermediate2 = new Position(intermediate1.RankValue + Math.Sign(newPosition.RankValue - originalPosition.RankValue), intermediate1.FileValue);
                }
                else
                {
                    intermediate1 = new Position(originalPosition.RankValue, originalPosition.FileValue + Math.Sign(newPosition.FileValue - originalPosition.FileValue));
                    intermediate2 = new Position(intermediate1.RankValue, intermediate1.FileValue + Math.Sign(newPosition.FileValue - originalPosition.FileValue));
                }

                // Add the intermediate positions
                path.Add(intermediate1);
                path.Add(intermediate2);

                // Add the new position
                path.Add(newPosition);
            }
            else
            {
                // Calculate the path for other pieces (assuming straight-line moves)
                int rankDifference = newPosition.RankValue - originalPosition.RankValue;
                int fileDifference = newPosition.FileValue - originalPosition.FileValue;

                int rankIncrement = Math.Sign(rankDifference);
                int fileIncrement = Math.Sign(fileDifference);

                int currentRank = originalPosition.RankValue;
                int currentFile = originalPosition.FileValue;

                while (currentRank != newPosition.RankValue || currentFile != newPosition.FileValue)
                {
                    currentRank += rankIncrement;
                    currentFile += fileIncrement;

                    path.Add(new Position(currentRank, currentFile));
                }
            }

            return path;
        }

        /// <summary>
        /// Gets the positions on the board.
        /// </summary>
        /// <returns>The positions.</returns>
        public Position[][] GetPositionsOnBoard()
        {
            Position[][] positions = new Position[8][];
            for (int i = 0; i < 8; ++i)
            {
                positions[i] = new Position[8];
                for (int j = 0; j < 8; ++j)
                {
                    positions[i][j] = new Position(j, i);
                }
            }

            return positions;
        }

        /// <summary>
        /// Gets the game phase.
        /// </summary>
        /// <returns>The game phase.</returns>
        public GamePhase GetGamePhase()
        {
            if (this.Board.ExecutedMoves.Count < 15)
            {
                return GamePhase.Opening;
            }
            else if (this.Board.GetPieceCount() < 11 && this.Board.ExecutedMoves.Count > 40)
            {
                return GamePhase.EndGame;
            }
            else
            {
                return GamePhase.MiddleGame;
            }
        }

        /// <summary>
        /// Get all moves.
        /// </summary>
        /// <returns>All moves.</returns>
        public List<string> GetMoves()
        {
            HashSet<string> moves = new HashSet<string>();
            for (int rank = 0; rank < 8; ++rank)
            {
                for (int file = 0; file < 8; ++file)
                {
                    var position = new Position(rank, file);
                    var piece = this.Board.GetPossiblePieceAt(position);
                    if (piece != null)
                    {
                        var list = piece.GetAvailableMovesToAnyColor(this.Board, position);
                        foreach (var m in list)
                        {
                            if (m.piece != null)
                            {
                                moves.Add(piece.ToString() + m.move.OriginalPosition.ToString() + m.move.NewPosition.ToString() + m.piece.ToString());
                            }
                            else
                            {
                                moves.Add(piece.ToString() + m.move.OriginalPosition.ToString() + m.move.NewPosition.ToString());
                            }
                        }
                    }
                }
            }

            return moves.ToList();
        }

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
            if (this.Board.EndGame?.EndgameType == EndgameType.Repetition)
            {
            }

            return this.Board.EndGame?.EndgameType == EndgameType.DrawDeclared
                ||
                this.Board.EndGame?.EndgameType == EndgameType.Checkmate
                ||
                this.Board.EndGame?.EndgameType == EndgameType.Stalemate
                ||
                this.Board.EndGame?.EndgameType == EndgameType.Repetition
                ||
                this.Board.EndGame?.EndgameType == EndgameType.InsufficientMaterial;
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
            var moves = this.Board.Moves();
            var isValid = moves.Any(x => x.ToString().Contains(move.ToString().Trim('{', '}').Trim('{', '}')));
            if (!isValid)
            {
            }

            return isValid;
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

                    var pos = new Position((short)(position.FileValue + j), (short)(position.RankValue + i));
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
                    var castle1 = new Position((short)(position.FileValue + 2), position.RankValue);
                    if (this.Board.IsValid(castle1))
                    {
                        positions.Add(castle1);
                    }

                    var castle2 = new Position((short)(position.FileValue - 2), position.RankValue);
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

                var pos = new Position(position.FileValue, (short)(position.RankValue + i));
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

                var pos = new Position((short)(position.FileValue + j), position.RankValue);
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

                    var pos = new Position((short)(position.FileValue + j), (short)(position.RankValue + i));
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

                var pos = new Position(position.FileValue, (short)(position.RankValue + i));
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

                var pos = new Position((short)(position.FileValue + j), position.RankValue);
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

                    var pos = new Position((short)(position.FileValue + j), (short)(position.RankValue + i));
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
                    if (i == 0 || j == 0)
                    {
                        continue;
                    }

                    var pos = new Position((short)(position.FileValue + j), (short)(position.RankValue + i));
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

                var pos = new Position(position.FileValue, (short)(position.RankValue + i));
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

                    var pos = new Position((short)(position.FileValue + j), (short)(position.RankValue + i));
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
            edge.PieceType = startPos.PieceType;
            edge.MoveType = moveInfo.MoveType;
            edge.CapturePieceType = moveInfo.CapturePieceType;
            edge.PromotionPieceType = moveInfo.PromotionPieceType;
            this.idToEdgeMap.AddOrUpdate(edge.Id, edge, (key, oldValue) => edge);
            this.graph.Edges.Add(edge);
            nodeFrom.Edges.Add(edge);
            nodeTo.Edges.Add(edge);
        }

        /// <summary>
        /// Populate the nodes.
        /// </summary>
        /// <param name="graph">The graph to populate.</param>
        /// <returns>The graph.</returns>
        public GapGraph PopulateNodes(GapGraph graph)
        {
            foreach (var node in graph.GapNodes)
            {
                var position = new Position((short)node.PositionX, (short)node.PositionY);
                var piece = this.Board[position];
                if (piece != null)
                {
                    node.Tag = piece.ToString();
                    switch (piece.ToString())
                    {
                        case "wp":
                        case "bp":
                            node.GapType = GapType.Pawn;
                            break;
                        case "wn":
                        case "bn":
                            node.GapType = GapType.Knight;
                            break;
                        case "wb":
                        case "bb":
                            node.GapType = GapType.Bishop;
                            break;
                        case "wr":
                        case "br":
                            node.GapType = GapType.Rook;
                            break;
                        case "wq":
                        case "bq":
                            node.GapType = GapType.Queen;
                            break;
                        case "wk":
                        case "bk":
                            node.GapType = GapType.King;
                            break;
                        default:
                            throw new InvalidOperationException("Invalid piece type");
                    }
                }
                else
                {
                    node.GapType = GapType.Empty;
                }
            }

            return graph;
        }

        private static (GapEdge Edge1, GapEdge Edge2) GetEdges(GapNode node1, GapNode node2, string move, bool isLegal, bool yourTurn)
        {
            move = move.ToLowerInvariant().Replace("o-o-o", "o3").Replace("o-o", "o2").Replace("{", string.Empty).Replace("}", string.Empty).Replace(" ", string.Empty).Replace("-", string.Empty);
            GapEdge gapEdge1 = new GapEdge()
            {
                Id = Guid.NewGuid(),
                Node = node1,
                Tag = new { Start = true, IsLegal = isLegal, YourTurn = yourTurn, Move = move },
            };
            node1.Edges.Add(gapEdge1);
            GapEdge gapEdge2 = new GapEdge()
            {
                Id = Guid.NewGuid(),
                Node = node2,
                Tag = new { Start = false, IsLegal = isLegal, YourTurn = yourTurn, Move = move },
            };
            node2.Edges.Add(gapEdge2);

            return (gapEdge1, gapEdge2);
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

                                    if (pawnPosition.RankValue == 0 || pawnPosition.RankValue == 7)
                                    {
                                        for (int l = 0; l < pieceTypes.Count; ++l)
                                        {
                                            this.AddToMap(key, (pawnPosition, MoveType.Promotion, null, pieceTypes[l].AsChar));
                                        }
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
