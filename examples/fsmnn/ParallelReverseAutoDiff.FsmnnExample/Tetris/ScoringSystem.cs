//------------------------------------------------------------------------------
// <copyright file="ScoringSystem.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// A scoring system.
    /// </summary>
    public class ScoringSystem
    {
        // Base points for various actions
        private const int SingleLinePoints = 100;
        private const int DoubleLinePoints = 300;
        private const int TripleLinePoints = 500;
        private const int TetrisPoints = 800;  // Clearing 4 lines at once
        private const int FlowerBombPoints = 150;
        private const int TSpinPoints = 400;
        private const int SoftDropPoints = 1;
        private const int HardDropPoints = 2;

        private int comboCounter;

        /// <summary>
        /// Gets the overall score.
        /// </summary>
        public int Score { get; private set; }

        /// <summary>
        /// Add to score for lines cleared.
        /// </summary>
        /// <param name="linesCleared">The number of lines cleared.</param>
        public void AddScoreForLinesCleared(int linesCleared)
        {
            this.comboCounter = (linesCleared > 0) ? this.comboCounter + 1 : 0;

            switch (linesCleared)
            {
                case 1:
                    this.Score += SingleLinePoints * this.comboCounter;
                    break;
                case 2:
                    this.Score += DoubleLinePoints * this.comboCounter;
                    break;
                case 3:
                    this.Score += TripleLinePoints * this.comboCounter;
                    break;
                case 4:
                    this.Score += TetrisPoints * this.comboCounter;
                    break;
            }
        }

        /// <summary>
        /// Add to the score for a flower bomb.
        /// </summary>
        public void AddScoreForFlowerBomb()
        {
            this.Score += FlowerBombPoints;
        }

        /// <summary>
        /// Add to the score for a T spin.
        /// </summary>
        public void AddScoreForTSpin()
        {
            this.Score += TSpinPoints;
        }

        /// <summary>
        /// Add to the score for a soft drop.
        /// </summary>
        /// <param name="cellsDropped">The number of cells dropped.</param>
        public void AddScoreForSoftDrop(int cellsDropped)
        {
            this.Score += SoftDropPoints * cellsDropped;
        }

        /// <summary>
        /// Add to the score for a hard drop.
        /// </summary>
        /// <param name="cellsDropped">The number of cells dropped.</param>
        public void AddScoreForHardDrop(int cellsDropped)
        {
            this.Score += HardDropPoints * cellsDropped;
        }

        /// <summary>
        /// Add to the score for a level multiplier.
        /// </summary>
        /// <param name="level">The level.</param>
        public void ApplyLevelMultiplier(int level)
        {
            this.Score *= level;
        }

        /// <summary>
        /// Resets the score.
        /// </summary>
        public void Reset()
        {
            this.Score = 0;
            this.comboCounter = 0;
        }
    }
}
