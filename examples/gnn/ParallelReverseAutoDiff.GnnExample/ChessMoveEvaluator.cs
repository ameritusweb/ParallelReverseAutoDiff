//------------------------------------------------------------------------------
// <copyright file="ChessMoveEvaluator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// Chess game evaluator.
    /// </summary>
    public class ChessMoveEvaluator
    {
        private Random rng;

        /// <summary>
        /// Initializes a new instance of the <see cref="ChessMoveEvaluator"/> class.
        /// </summary>
        public ChessMoveEvaluator()
        {
            this.rng = new Random(Guid.NewGuid().GetHashCode());
        }

        /// <summary>
        /// Gets a random number generator.
        /// </summary>
        public Random Random
        {
            get
            {
                return this.rng;
            }
        }

        /// <summary>
        /// Compute the reward.
        /// </summary>
        /// <param name="state">The game state.</param>
        /// <param name="color">The piece color.</param>
        /// <param name="move">The move.</param>
        /// <returns>The computed reward.</returns>
        public double ComputeReward(GameState state, PieceColor color, Move move)
        {
            // Apply the move to the game state
            state.Board.Move(move);

            double score = this.GetMoveScore(move);

            // Compute the feature values of the new game state
            double[] featureValues = this.GetFeatureValues(state, color);

            // Compute the reward based on the feature values
            var reward = score + featureValues.Sum();

            // Undo the move to restore the game state to its original state
            state.Board.Cancel();

            return reward;
        }

        /// <summary>
        /// Gets the score of a move.
        /// </summary>
        /// <param name="move">The move.</param>
        /// <returns>The move score.</returns>
        public double GetMoveScore(Move move)
        {
            double score = 0;

            if (move.Parameter != null)
            {
                if (move.Parameter is MoveCastle)
                {
                    score += 50d;
                }
                else if (move.Parameter is MovePromotion)
                {
                    score += 200d;
                }
                else if (move.Parameter is MoveEnPassant)
                {
                    score += 10d;
                }
            }

            if (move.CapturedPiece != null && move.CapturedPiece.MaterialValue > move.Piece.MaterialValue)
            {
                score += 50d;
            }

            if (move.IsMate)
            {
                score += 500d;
            }

            return score;
        }

        /// <summary>
        /// Gets the feature values of a game state.
        /// </summary>
        /// <param name="state">The game state.</param>
        /// <param name="color">The piece color.</param>
        /// <returns>The feature values.</returns>
        public double[] GetFeatureValues(GameState state, PieceColor color)
        {
            double[] featureValues = new double[16];
            EvaluationTable evaluationTable = new EvaluationTable();
            evaluationTable.SetTable(state);
            evaluationTable.PassMessages();
            short[][] squareControl = this.CalculateSquareControl(state, color, evaluationTable);
            var lastmove = state.Board.ExecutedMoves.LastOrDefault();
            var lastOpponentMove = state.Board.ExecutedMoves.SkipLast(1).LastOrDefault();
            var allMoves = state.GetAllMoves();
            featureValues[0] = this.GetControlOfCenter(state, color);
            featureValues[1] = this.GetOpponentControlOfCenter(state, color);
            featureValues[4] = this.GetKingSafetyScore(state, color, allMoves);
            featureValues[5] = this.CountDefendingPieces(state, color);
            featureValues[6] = this.CountDefendedPieces(state, color);
            featureValues[7] = this.CountCheckmate(state, color);
            featureValues[8] = this.GetPieceSafety(state, color, squareControl, lastmove, allMoves);
            featureValues[9] = this.GetCheckmateSafety(state, color, allMoves);
            featureValues[10] = this.GetPawnStructureScore(state, color);
            featureValues[11] = this.GetPieceDevelopment(state, color);
            if (featureValues[8] >= 0d)
            {
                featureValues[12] = this.GetTacticScore(state, color, allMoves);
                featureValues[3] = this.GetMobility(state, color, allMoves);
                featureValues[2] = this.GetMaterialAdvantage(state, color);
            }
            else
            {
                featureValues[12] = 0d;
                featureValues[3] = 0d;
                featureValues[2] = 0d;
            }

            featureValues[13] = this.GetCaptureScore(lastmove, lastOpponentMove, squareControl, featureValues[2] >= 0d);
            featureValues[14] = evaluationTable.GetStackedScore(color);
            featureValues[15] = this.GetTacticScore(state, color.OppositeColor(), allMoves) * -1 / 2d;
            return featureValues;
        }

        /// <summary>
        /// Gets a score for tactical advantage.
        /// </summary>
        /// <param name="state">The game state.</param>
        /// <param name="color">The piece color.</param>
        /// <param name="allMoves">All moves.</param>
        /// <returns>The tactic score.</returns>
        public double GetTacticScore(GameState state, PieceColor color, List<Move> allMoves)
        {
            double tacticScore = 0.0d;
            double attackScore = 0.0d;

            // Assign weights to each tactic
            const double PinWeight = 1.0d;
            const double ForkWeight = 1.0d;
            const double AttackWeight = 0.5d;

            // Iterate through all the pieces of the specified color
            var piecePositions = state.Board.GetPositions(color);

            foreach (var position in piecePositions)
            {
                var piece = state.Board[position] ?? throw new InvalidOperationException("Piece must not be null.");

                // Check for pins on the king and queen by sliding pieces (bishop, rook, queen)
                if (piece.Type == PieceType.Bishop || piece.Type == PieceType.Rook || piece.Type == PieceType.Queen)
                {
                    if (this.IsPiecePinning(state, position, color))
                    {
                        tacticScore += PinWeight;
                    }
                }

                // Check for forks
                if (this.IsPieceForking(state, position, color, allMoves))
                {
                    tacticScore += ForkWeight;
                }

                // Check for attacks
                if (this.IsPieceAttacking(state, position, color, allMoves))
                {
                    attackScore += AttackWeight;
                }
            }

            // Normalize the score
            const double MaxScore = 2d; // Assuming a maximum of 1 pin and 1 fork
            double normalizedScore = (tacticScore + (Math.Pow(2d, attackScore) / MaxScore)) * MaxScore;

            return normalizedScore;
        }

        /// <summary>
        /// Get a score for piece development.
        /// </summary>
        /// <param name="state">The game state.</param>
        /// <param name="color">The piece color.</param>
        /// <returns>The piece development score.</returns>
        public double GetPieceDevelopment(GameState state, PieceColor color)
        {
            double undevelopedPieceScore = 0.0;

            var board = state.Board;

            // Assign weights to each piece
            const double PawnWeight = 0.4;
            const double KnightWeight = 0.5;
            const double BishopWeight = 0.5;
            const double RookWeight = 0.25;
            const double QueenWeight = 0.25;

            for (int rank = 0; rank < 8; rank++)
            {
                for (int file = 0; file < 8; file++)
                {
                    var position = new Position(rank, file);
                    var piece = board[position];
                    if (piece != null && piece.Color == color)
                    {
                        bool isUndeveloped = false;

                        switch (piece.Type.Name)
                        {
                            case nameof(PieceType.Knight):
                                isUndeveloped = (rank == (color == PieceColor.White ? 0 : 7)) && (file == 1 || file == 6);
                                if (isUndeveloped)
                                {
                                    undevelopedPieceScore += KnightWeight;
                                }

                                break;

                            case nameof(PieceType.Pawn):
                                isUndeveloped = rank == (color == PieceColor.White ? 1 : 6);
                                if (isUndeveloped)
                                {
                                    undevelopedPieceScore += PawnWeight;
                                }

                                break;

                            case nameof(PieceType.Bishop):
                                isUndeveloped = (rank == (color == PieceColor.White ? 0 : 7)) && (file == 2 || file == 5);
                                if (isUndeveloped)
                                {
                                    undevelopedPieceScore += BishopWeight;
                                }

                                break;

                            case nameof(PieceType.Rook):
                                isUndeveloped = (rank == (color == PieceColor.White ? 0 : 7)) && (file == 0 || file == 7);
                                if (isUndeveloped)
                                {
                                    undevelopedPieceScore += RookWeight;
                                }

                                break;

                            case nameof(PieceType.Queen):
                                isUndeveloped = (rank == (color == PieceColor.White ? 0 : 7)) && (file == 3);
                                if (isUndeveloped)
                                {
                                    undevelopedPieceScore += QueenWeight;
                                }

                                break;
                        }
                    }
                }
            }

            // Normalize the score
            const double MaxUndevelopedPieceScore = 4.7; // Assuming all Knights, Bishops, and Queen are undeveloped
            double normalizedScore = (MaxUndevelopedPieceScore - undevelopedPieceScore) / MaxUndevelopedPieceScore;

            return normalizedScore;
        }

        /// <summary>
        /// Gets the pawn structure score.
        /// </summary>
        /// <param name="state">The game state.</param>
        /// <param name="color">The piece color.</param>
        /// <returns>The pawn structure score.</returns>
        public double GetPawnStructureScore(GameState state, PieceColor color)
        {
            (int doubledPawns, int pawnIslands, int longPawnChains)
                = this.AnalyzePawnStructure(state, color);

            // Define weights for each feature
            const double DoubledPawnWeight = -1.0;
            const double PawnIslandWeight = -0.5;
            const double LongPawnChainWeight = 1.0;

            // Calculate the weighted sum of the features
            double score = (DoubledPawnWeight * doubledPawns) +
                           (PawnIslandWeight * pawnIslands) +
                           (LongPawnChainWeight * longPawnChains);

            // Normalize the score
            const double MaxScore = 8; // Assuming a maximum of 8 pawn islands, doubled pawns, and long pawn chains
            double normalizedScore = (score + MaxScore) / (2 * MaxScore);

            // Return the normalized score
            return normalizedScore;
        }

        /// <summary>
        /// Compute the reward.
        /// </summary>
        /// <param name="state">The game state.</param>
        /// <param name="color">The piece color.</param>
        /// <returns>The rwward.</returns>
        public double ComputeReward(GameState state, PieceColor color)
        {
            // Compute the feature values of the new game state
            double[] featureValues = this.GetFeatureValues(state, color);

            // Compute the reward based on the feature values
            var reward = featureValues.Sum();

            return reward;
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

        private (int DoubledPawns, int PawnIslands, int LongPawnChains) AnalyzePawnStructure(GameState state, PieceColor color)
        {
            var board = state.Board;
            int[] pawnFiles = new int[8];
            int[] pawnRanks = new int[8];
            int longestChain = 0;
            int doubledPawnCount = 0;
            int pawnIslandCount = 0;

            // Count pawns on each file and rank
            for (int rank = 0; rank < 8; rank++)
            {
                for (int file = 0; file < 8; file++)
                {
                    var position = new Position(rank, file);
                    var piece = board[position];
                    if (piece != null && piece.Color == color && piece.Type == PieceType.Pawn)
                    {
                        pawnFiles[file]++;
                        pawnRanks[rank]++;

                        // Find the longest pawn chain for each starting pawn
                        int currentChain = this.FindLongestPawnChain(board, color, position, new HashSet<Position>()) - 1;
                        if (currentChain > longestChain && currentChain >= 2)
                        {
                            longestChain = currentChain;
                        }
                    }
                }
            }

            for (int file = 0; file < 8; file++)
            {
                if (pawnFiles[file] > 1)
                {
                    doubledPawnCount += pawnFiles[file] - 1;
                }

                if (pawnFiles[file] > 0)
                {
                    bool leftFileEmpty = (file == 0) || (pawnFiles[file - 1] == 0);
                    bool rightFileEmpty = (file == 7) || (pawnFiles[file + 1] == 0);

                    if (leftFileEmpty && rightFileEmpty)
                    {
                        pawnIslandCount++;
                    }
                }
            }

            return (doubledPawnCount, pawnIslandCount, longestChain);
        }

        private int FindLongestPawnChain(ChessBoard board, PieceColor color, Position position, HashSet<Position> visited)
        {
            if (!board.IsValid(position) || visited.Contains(position))
            {
                return 0;
            }

            visited.Add(position);

            var piece = board[position];
            if (piece == null || piece.Color != color || piece.Type != PieceType.Pawn)
            {
                return 0;
            }

            int maxChainLength = 0;

            int[] rankOffsets = { -1, 1 };
            int[] fileOffsets = { -1, 1 };

            foreach (int rankOffset in rankOffsets)
            {
                foreach (int fileOffset in fileOffsets)
                {
                    Position newPosition = position.MoveBy(rankOffset, fileOffset);
                    int chainLength = this.FindLongestPawnChain(board, color, newPosition, visited);
                    maxChainLength = Math.Max(maxChainLength, chainLength);
                }
            }

            visited.Remove(position);

            return 1 + maxChainLength;
        }

        private bool IsPieceForking(GameState gameState, Position piecePosition, PieceColor color, List<Move> allMoves)
        {
            Piece? piece = gameState.Board[piecePosition];
            if (piece == null || piece.Color != color)
            {
                return false;
            }

            var opponentColor = color == PieceColor.White ? PieceColor.Black : PieceColor.White;
            var moves = gameState.GetAllMovesForPositionAndColor(piecePosition, color, allMoves);

            int attackedKingCount = 0;
            int attackedPieceCount = 0;

            foreach (var move in moves)
            {
                var targetPosition = move.NewPosition;
                var targetPiece = gameState.Board[targetPosition];

                if (targetPiece != null && targetPiece.Color == opponentColor)
                {
                    if (targetPiece.Type == PieceType.King)
                    {
                        attackedKingCount++;
                    }
                    else
                    {
                        attackedPieceCount++;
                    }
                }
            }

            return attackedKingCount >= 1 && attackedPieceCount >= 1;
        }

        private bool IsPieceAttacking(GameState gameState, Position piecePosition, PieceColor color, List<Move> allMoves)
        {
            Piece? piece = gameState.Board[piecePosition];
            if (piece == null || piece.Color != color)
            {
                return false;
            }

            var opponentColor = color == PieceColor.White ? PieceColor.Black : PieceColor.White;
            var moves = gameState.GetAllMovesForPositionAndColor(piecePosition, color, allMoves);

            int attackedPieceCount = 0;

            foreach (var move in moves)
            {
                var targetPosition = move.NewPosition;
                var targetPiece = gameState.Board[targetPosition];

                if (targetPiece != null && targetPiece.Color == opponentColor)
                {
                    attackedPieceCount++;
                }
            }

            return attackedPieceCount >= 1;
        }

        private bool IsPiecePinning(GameState gameState, Position piecePosition, PieceColor color)
        {
            Piece? piece = gameState.Board[piecePosition];
            if (piece == null || piece.Color != color)
            {
                return false;
            }

            var opponentColor = color == PieceColor.White ? PieceColor.Black : PieceColor.White;

            int[][] pieceOffsets;

            switch (piece.Type.Name)
            {
                case nameof(PieceType.Bishop):
                    pieceOffsets = new[]
                    {
                        new[] { -1, -1 },
                        new[] { -1, 1 },
                        new[] { 1, -1 },
                        new[] { 1, 1 },
                    };
                    break;
                case nameof(PieceType.Rook):
                    pieceOffsets = new[]
                    {
                        new[] { -1, 0 },
                        new[] { 1, 0 },
                        new[] { 0, -1 },
                        new[] { 0, 1 },
                    };
                    break;
                case nameof(PieceType.Queen):
                    pieceOffsets = new[]
                    {
                        new[] { -1, -1 },
                        new[] { -1, 0 },
                        new[] { -1, 1 },
                        new[] { 0, -1 },
                        new[] { 0, 1 },
                        new[] { 1, -1 },
                        new[] { 1, 0 },
                        new[] { 1, 1 },
                    };
                    break;
                default:
                    return false;
            }

            foreach (int[] offset in pieceOffsets)
            {
                int rankOffset = offset[0];
                int fileOffset = offset[1];

                Position currentPosition = piecePosition.MoveBy(rankOffset, fileOffset);
                bool foundBlockingPiece = false;

                while (gameState.Board.IsValid(currentPosition))
                {
                    Piece? currentPiece = gameState.Board[currentPosition];

                    if (currentPiece != null)
                    {
                        if (currentPiece.Color == color)
                        {
                            break;
                        }
                        else if ((currentPiece.Type == PieceType.King || currentPiece.Type == PieceType.Queen) && currentPiece.Color == opponentColor && foundBlockingPiece)
                        {
                            return true;
                        }
                        else
                        {
                            if (foundBlockingPiece)
                            {
                                break;
                            }

                            foundBlockingPiece = true;
                        }
                    }

                    currentPosition = currentPosition.MoveBy(rankOffset, fileOffset);
                }
            }

            return false;
        }

        private double CountCheckmate(GameState state, PieceColor color)
        {
            if (state.Board.IsEndGameCheckmate
                &&
                state.Board.EndGame?.WonSide == color)
            {
                return 8.0d;
            }
            else if (state.Board.IsEndGameCheckmate
                &&
                state.Board.EndGame?.WonSide == color.OppositeColor())
            {
                return -8.0d;
            }

            return 0.0d;
        }

        private double CountDefendingPieces(GameState state, PieceColor color)
        {
            int count = 0;
            foreach (Tuple<Piece, Position> piece in state.Board.GetPiecesAndTheirPositions(color))
            {
                try
                {
                    var moveAndPieceList = piece.Item1.GetAvailableMovesToTheSameColor(state.Board, piece.Item2);
                    foreach (var moveAndPiece in moveAndPieceList)
                    {
                        if (piece.Item1.Type == PieceType.Pawn)
                        {
                            count += 4;
                        }
                        else if (piece.Item1.Type == PieceType.Knight || piece.Item1.Type == PieceType.Bishop)
                        {
                            count += 3;
                        }
                        else if (piece.Item1.Type == PieceType.Rook)
                        {
                            count += 2;
                        }
                        else if (piece.Item1.Type == PieceType.Queen)
                        {
                            count += 1;
                        }
                    }
                }
                catch (Exception)
                {
                }
            }

            return Math.Tanh(count / 16d);
        }

        private double CountDefendedPieces(GameState state, PieceColor color)
        {
            int count = 0;
            foreach (Tuple<Piece, Position> piece in state.Board.GetPiecesAndTheirPositions(color))
            {
                try
                {
                    var moveAndPieceList = piece.Item1.GetAvailableMovesToTheSameColor(state.Board, piece.Item2);
                    foreach (var moveAndPiece in moveAndPieceList)
                    {
                        var p = moveAndPiece.Item2;
                        if (p.Type == PieceType.Pawn)
                        {
                            count += 1;
                        }
                        else if (p.Type == PieceType.Knight)
                        {
                            count += 3;
                        }
                        else if (p.Type == PieceType.Bishop)
                        {
                            count += 3;
                        }
                        else if (p.Type == PieceType.Rook)
                        {
                            count += 5;
                        }
                        else if (p.Type == PieceType.Queen)
                        {
                            count += 9;
                        }
                    }
                }
                catch (Exception)
                {
                }
            }

            return Math.Tanh(count / 16d);
        }

        private double GetControlOfCenter(GameState state, PieceColor color)
        {
            double control = 0.0;
            for (int i = 3; i <= 4; i++)
            {
                for (int j = 3; j <= 4; j++)
                {
                    Position square = new Position((short)i, (short)j);
                    if (state.GetPieceAt(square)?.Color == color)
                    {
                        control += 1.0;
                    }
                }
            }

            if (control == 0.0d)
            {
                return -1d;
            }

            return Math.Tanh(control);
        }

        private double GetOpponentControlOfCenter(GameState state, PieceColor color)
        {
            double control = 0.0;
            PieceColor opponent = color == PieceColor.White ? PieceColor.Black : PieceColor.White;
            for (int i = 3; i <= 4; i++)
            {
                for (int j = 3; j <= 4; j++)
                {
                    Position square = new Position((short)i, (short)j);
                    if (state.GetPieceAt(square)?.Color == opponent)
                    {
                        control += 1.0;
                    }
                }
            }

            if (control == 0.0d)
            {
                return 1d;
            }

            return -1d * Math.Tanh(control);
        }

        private double GetPieceSafety(GameState state, PieceColor color, short[][] squareControl, Move? lastMove, List<Move> allMoves)
        {
            double safetyScore = 0d;
            var movesForOpponent = state.GetAllMovesForColor(color.OppositeColor(), allMoves);
            foreach (var move in movesForOpponent.Where(x => x.CapturedPiece != null))
            {
                var piece = move.Piece.MaterialValue;
                if (move.CapturedPiece!.MaterialValue > piece)
                {
                    if (lastMove != null
                        &&
                        lastMove.CapturedPiece != null
                        &&
                        lastMove.NewPosition.ToString() == move.NewPosition.ToString()
                        &&
                        lastMove.CapturedPiece.MaterialValue >= move.CapturedPiece.MaterialValue)
                    {
                        safetyScore -= 0d;
                    }
                    else
                    {
                        safetyScore -= move.CapturedPiece!.MaterialValue - piece;
                    }
                }
                else if (squareControl[move.NewPosition.Y][move.NewPosition.X] < 0)
                {
                    if (lastMove != null
                        &&
                        lastMove.CapturedPiece != null
                        &&
                        lastMove.NewPosition.ToString() == move.NewPosition.ToString()
                        &&
                        lastMove.CapturedPiece.MaterialValue >= move.CapturedPiece.MaterialValue)
                    {
                        safetyScore -= move.Piece.MaterialValue / 9d;
                    }
                    else
                    {
                        safetyScore -= move.CapturedPiece.MaterialValue;
                    }
                }
            }

            return safetyScore;
        }

        private double GetCheckmateSafety(GameState state, PieceColor color, List<Move> allMoves)
        {
            double safetyScore = 0d;
            var movesForOpponent = state.GetAllMovesForColor(color.OppositeColor(), allMoves);
            foreach (var move in movesForOpponent.Where(x => x.IsMate))
            {
                safetyScore -= 4d;
            }

            return safetyScore;
        }

        private double GetMaterialAdvantage(GameState state, PieceColor color)
        {
            // Determine the material value of each player's pieces
            double currentPlayerMaterial = 0.0;
            double opponentMaterial = 0.0;
            foreach (Piece piece in state.Board.GetPieces())
            {
                if (piece.Color == color)
                {
                    currentPlayerMaterial += piece.MaterialValue;
                }
                else
                {
                    opponentMaterial += piece.MaterialValue;
                }
            }

            // Return a score between -1 and 1 based on the difference in material
            double materialDifference = currentPlayerMaterial - opponentMaterial;
            return Math.Tanh(materialDifference / 20.0) * 2d * Math.Abs(materialDifference);
        }

        private double GetMobility(GameState state, PieceColor color, List<Move> allMoves)
        {
            // Count the number of legal moves for both colors
            int playerLegalMoves = allMoves.Count(move => move.Piece.Color == color);
            int opponentLegalMoves = allMoves.Count(move => move.Piece.Color != color);

            // Normalize the mobility scores by the maximum possible number of legal moves
            double maxLegalMoves = 2 * state.BoardSize * state.BoardSize;
            double normalizedPlayerLegalMoves = playerLegalMoves / maxLegalMoves;
            double normalizedOpponentLegalMoves = opponentLegalMoves / maxLegalMoves;

            // Return the difference in normalized legal moves as a score
            return 10d * (normalizedPlayerLegalMoves - normalizedOpponentLegalMoves);
        }

        private double GetKingSafetyScore(GameState state, PieceColor color, List<Move> allMoves)
        {
            // Calculate the number of safe squares around the player's king
            int safeSquares = 0;
            Position? kingSquare = state.GetKingSquare(color);

            if (kingSquare == null)
            {
                return 0.0;
            }

            // Check if any of the surrounding squares are under attack by the opponent
            Position[] surroundingSquares = kingSquare.Value.GetAdjacentSquares().ToArray();
            var opponentColor = color == PieceColor.White ? PieceColor.Black : PieceColor.White;
            foreach (Position square in surroundingSquares)
            {
                // If the square is occupied by a blocking piece or is empty, it is potentially safe
                if (state.Board.IsSquareEmpty(square) || state.GetPieceAt(square)?.Color == color)
                {
                    // Check if any of the opponent's pieces are attacking this square
                    bool squareUnderAttack = false;
                    foreach (Piece piece in state.Board.GetPieces(opponentColor))
                    {
                        if (piece.CanAttack(square, state.Board, allMoves.ToArray()))
                        {
                            squareUnderAttack = true;
                            break;
                        }
                    }

                    if (!squareUnderAttack)
                    {
                        safeSquares++;
                    }
                }
            }

            var kingSafety = safeSquares / 8.0; // Normalize by the maximum number of safe squares
            if (kingSquare.HasValue)
            {
                if (kingSquare.Value.Y != 0 && kingSquare.Value.Y != 7)
                {
                    kingSafety -= 1;
                }
            }

            return kingSafety;
        }

        private double GetCaptureScore(Move? lastMove, Move? lastOpponentMove, short[][] squareControl, bool hasMaterialAdvantage)
        {
            var captureScore = 0d;
            var wasCapture = lastOpponentMove != null && lastOpponentMove.CapturedPiece != null;
            if (lastMove != null && lastMove.CapturedPiece != null)
            {
                var value = lastMove.CapturedPiece.MaterialValue;
                var endPosition = lastMove.NewPosition;
                var control = squareControl[endPosition.Y][endPosition.X];
                var startPosition = lastMove.OriginalPosition;
                var startControl = squareControl[startPosition.Y][startPosition.X];
                if (control > 0)
                {
                    captureScore = value;
                }
                else if (control == 0)
                {
                    if (lastMove.Piece.MaterialValue < lastMove.CapturedPiece.MaterialValue)
                    {
                        captureScore = value;
                    }
                    else if (lastMove.Piece.MaterialValue == lastMove.CapturedPiece.MaterialValue)
                    {
                        if (hasMaterialAdvantage)
                        {
                            captureScore = 1d;
                        }
                    }
                }
                else
                {
                    if (lastMove.Piece.MaterialValue < lastMove.CapturedPiece.MaterialValue)
                    {
                        if (startControl <= 0)
                        {
                            captureScore = value * 2d;
                        }
                        else
                        {
                            captureScore = value;
                        }
                    }
                }

                if (wasCapture && captureScore == 0d && lastOpponentMove != null && lastOpponentMove.NewPosition.ToString() == lastMove.NewPosition.ToString() && lastMove.CapturedPiece.MaterialValue >= lastMove.Piece.MaterialValue)
                {
                    captureScore += value;
                }
            }

            return captureScore;
        }
    }
}
