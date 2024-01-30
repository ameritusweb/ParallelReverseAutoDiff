//------------------------------------------------------------------------------
// <copyright file="ProjectionGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.Drawing;
    using Chess;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths;

    /// <summary>
    /// A projection generator.
    /// </summary>
    public class ProjectionGenerator
    {
        private GameState gameState;

        private ChessBoardLoader loader;

        private Random rand = new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        /// Initializes a new instance of the <see cref="ProjectionGenerator"/> class.
        /// </summary>
        public ProjectionGenerator()
        {
            this.gameState = new GameState();
            this.loader = new ChessBoardLoader();
        }

        /// <summary>
        /// Generates a projection.
        /// </summary>
        /// <returns>Returns a task.</returns>
        public async Task Generate()
        {
            int total = this.loader.GetTotal();
            var r = this.rand.Next(total);
            var moves = this.loader.LoadMoves(r);
            var name = this.loader.GetFileName(r).Replace(".pgn", string.Empty);

            try
            {
                foreach ((Move move, Move? nextmove) in moves.WithNext())
                {
                    this.gameState.Board.Move(move);
                    if (nextmove != null)
                    {
                        List<Matrix> matrices = new List<Matrix>();
                        Matrix pieces = new Matrix(8, 8);
                        matrices.Add(pieces);
                        var gamePhase = this.gameState.GetGamePhase();
                        var allmoves = this.gameState.GetMoves();
                        var legalbothmoves = this.gameState.GetAllMoves();
                        var legalmoves = this.gameState.Board.Moves().ToList();
                        var turn = this.gameState.Board.Turn;
                        var fen = this.gameState.Board.ToFen();
                        EvaluationTable evaluationTable = new EvaluationTable();
                        evaluationTable.SetTable(this.gameState);
                        evaluationTable.PassMessages();
                        short[][] squareControlWhite = this.CalculateSquareControl(this.gameState, PieceColor.White, evaluationTable);
                        StockfishReader reader = new StockfishReader();
                        var score = await reader.ReadBestMoveScoreAsync(this.gameState);
                        for (int i = 0; i < 8; ++i)
                        {
                            for (int j = 0; j < 8; ++j)
                            {
                                Position pos = new Position(i, j);
                                var piece = this.gameState.Board[pos];
                                if (piece != null)
                                {
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private short[][] CalculateSquareControl(GameState state, PieceColor color, EvaluationTable table)
        {
            short[][] squareControl = new short[8][];
            for (int i = 0; i < squareControl.Length; ++i)
            {
                squareControl[i] = new short[8];
            }

            var positions = state.Board.GetPiecesAndTheirPositions(color);
            foreach (var position in positions)
            {
                var piece = position.Item1;
                var moves = piece.GetSquareControl(state.Board, position.Item2);
                foreach (var move in moves)
                {
                    var endPosition = move.Item1.NewPosition;
                    squareControl[endPosition.Y][endPosition.X]++;
                }
            }

            var opositions = state.Board.GetPiecesAndTheirPositions(color.OppositeColor());
            foreach (var position in opositions)
            {
                var piece = position.Item1;
                var moves = piece.GetSquareControl(state.Board, position.Item2);
                foreach (var move in moves)
                {
                    var endPosition = move.Item1.NewPosition;
                    squareControl[endPosition.Y][endPosition.X]--;
                }
            }

            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; ++j)
                {
                    var position = new Position(i, j);
                    var control = (short)table.GetSquareControl(position, color);

                    squareControl[position.Y][position.X] += control;

                    var ocontrol = (short)table.GetSquareControl(position, color.OppositeColor());

                    squareControl[position.Y][position.X] -= ocontrol;
                }
            }

            return squareControl;
        }
    }
}