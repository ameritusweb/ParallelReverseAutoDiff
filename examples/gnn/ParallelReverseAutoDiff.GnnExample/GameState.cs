namespace RandomForestTest
{
    using Chess;
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;

    public class GameState
    {
        public ChessBoard Board { get; set; }
        public int BoardSize { get; set; } = 8;

        private Random rng;
        private ConcurrentDictionary<(Position, char), List<(Move, char)>> positionToPossibleMoveMap;
        private ConcurrentDictionary<Position, GNNNode> positionToPieceMap;
        private ConcurrentDictionary<Position, GNNNode> positionToSquareMap;
        private GNNGraph graph;

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
                        var moveAndPieceList = pieceAndPosition.piece.GetAvailableMovesToAnyColor(Board, pieceAndPosition.position);
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

        public GameState()
        {
            this.Board = new ChessBoard();
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Move, char)>>();
            this.positionToPieceMap = new ConcurrentDictionary<Position, GNNNode>();
            this.positionToSquareMap = new ConcurrentDictionary<Position, GNNNode>();
            this.weightedGraph = new GNNWeightedGraph();
            this.BuildGraph();
        }

        public GameState(ChessBoard board)
        {
            this.Board = board;
            this.rng = new Random(DateTime.UtcNow.Millisecond);
            this.positionToPossibleMoveMap = new ConcurrentDictionary<(Position, char), List<(Move, char)>>();
            this.positionToPieceMap = new ConcurrentDictionary<Position, GNNNode>();
            this.positionToSquareMap = new ConcurrentDictionary<Position, GNNNode>();
            this.weightedGraph = new GNNWeightedGraph();
            this.BuildGraph();
        }

        public Position? GetKingSquare(PieceColor color)
        {
            return color == PieceColor.White ? Board.WhiteKing : Board.BlackKing;
        }

        public bool IsSquareEmpty(Position position)
        {
            return Board.pieces[position.X, position.Y] == null;
        }

        public IEnumerable<Piece> GetPieces()
        {
            List<Piece> pieces = new List<Piece>();
            for (int i = 0; i < BoardSize; i++)
            {
                for (int j = 0; j < BoardSize; ++j)
                {
                    if (Board.pieces[i, j] != null)
                    {
                        var piece = Board.pieces[i, j];
                        if (piece != null)
                        {
                            pieces.Add(piece);
                        }
                    }
                }
            }
            return pieces;
        }

        public IEnumerable<Piece> GetPieces(PieceColor color)
        {
            List<Piece> pieces = new List<Piece>();
            for (int i = 0; i < BoardSize; i++)
            {
                for (int j = 0; j < BoardSize; ++j)
                {
                    if (Board.pieces[i, j]?.Color == color)
                    {
                        var piece = Board.pieces[i, j];
                        if (piece != null)
                        {
                            pieces.Add(piece);
                        }
                    }
                }
            }
            return pieces;
        }

        public Piece? GetPieceAt(Position position)
        {
            return Board[position];
        }

        public Piece? GetPieceAt(int position)
        {
            return Board[new Position() { X = (short)(position / 8), Y = (short)(position % 8) }];
        }

        public IEnumerable<Position> GetPositions(PieceColor color)
        {
            List<Position> positions = new List<Position>();
            for (int i = 0; i < BoardSize; i++)
            {
                for (int j = 0; j < BoardSize; ++j)
                {
                    if (Board.pieces[i, j]?.Color == color)
                    {
                        var piece = Board.pieces[i, j];
                        if (piece != null)
                        {
                            positions.Add(new Position(i, j));
                        }
                    }
                }
            }
            return positions;
        }

        public List<Move> GetAllMoves()
        {
            return Board.Moves(false, true, false).ToList();
        }

        public List<Move> GetAllMovesForColor(PieceColor color)
        {
            return Board.Moves(false, true, false)
                .Where(x => x.Piece.Color == color)
                .ToList();
        }

        public List<Move> GetAllMovesForColorAndType(PieceColor color, PieceType type)
        {
            return Board.Moves(false, true, false)
                .Where(x => x.Piece.Color == color && x.Piece.Type == type)
                .ToList();
        }

        public List<Move> GetLegalMoves(int position)
        {
            var pos = new Position() { X = (short)(position / 8), Y = (short)(position % 8) };
            return Board.Moves().Where(x => x.OriginalPosition == pos).ToList();
        }

        public List<Move> GetLegalMoves(Position position)
        {
            return Board.Moves().Where(x => x.OriginalPosition == position).ToList();
        }

        public List<Move> GetLegalMovesForPositionAndColor(Position position, PieceColor color)
        {
            return Board.Moves().Where(x => x.OriginalPosition == position && x.Piece.Color == color).ToList();
        }

        public List<Move> GetAllMovesForPositionAndColor(Position position, PieceColor color)
        {
            return Board.Moves(false, true, false).Where(x => x.OriginalPosition == position && x.Piece.Color == color).ToList();
        }

        public Move MoveRandomly()
        {
            var moves = Board.Moves();
            var move = rng.Next() % moves.Length;
            Board.Move(moves[move]);
            return moves[move];
        }

        public Move MoveRandomly(List<Move> moves)
        {
            var move = rng.Next() % moves.Count;
            Board.Move(moves[move]);
            return moves[move];
        }

        public List<Move> LastMoves(int numOfMoves)
        {
            return Board.ExecutedMoves.TakeLast(numOfMoves).ToList();
        }

        public bool IsTerminal()
        {
            return Board.EndGame?.EndgameType == EndgameType.Checkmate || Board.EndGame?.EndgameType == EndgameType.Stalemate;
        }

        public bool IsGameOver()
        {
            return Board.EndGame?.EndgameType == EndgameType.DrawDeclared || Board.EndGame?.EndgameType == EndgameType.Checkmate || Board.EndGame?.EndgameType == EndgameType.Stalemate;
        }

        public bool IsValid(int rank, int file)
        {
            return Board.IsValid(new Position(rank, file));
        }

        public bool IsValidMove(Move move)
        {
            return Board.Moves().Any(x => x.ToString() == move.ToString());
        }
    }

}
