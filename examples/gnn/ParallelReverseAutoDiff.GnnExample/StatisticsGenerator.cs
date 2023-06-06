//------------------------------------------------------------------------------
// <copyright file="StatisticsGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    /// <summary>
    /// Initializes a new instance of the <see cref="StatisticsGenerator"/> class.
    /// </summary>
    public class StatisticsGenerator
    {
        private ChessBoardLoader loader = new ChessBoardLoader();

        /// <summary>
        /// Generates the statistics.
        /// </summary>
        public void Generate()
        {
            GameState gameState = new GameState();
            var moves = this.loader.LoadMoves(0);
        }
    }
}
