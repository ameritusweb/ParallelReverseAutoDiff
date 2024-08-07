﻿//------------------------------------------------------------------------------
// <copyright file="TileGrid.cs" author="ameritusweb" date="5/2/2024">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD;

    /// <summary>
    /// A tile grid.
    /// </summary>
    public class TileGrid : TileGridBase
    {
        private List<MazeComputationGraph> mazeComputationGraphs = new List<MazeComputationGraph>();
        private List<MazeComputationGraph> moreGraphs = new List<MazeComputationGraph>();
        private MazeNetwork mazeNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="TileGrid"/> class.
        /// </summary>
        /// <param name="mazeNetwork">The maze network.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="targets">The targets.</param>
        public TileGrid(MazeNetwork mazeNetwork, List<Matrix> inputs, List<double> targets)
            : base(inputs)
        {
            this.mazeNetwork = mazeNetwork;
            this.Inputs = inputs;
            this.Targets = targets;
        }

        /// <summary>
        /// Run time steps.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task RunTimeSteps()
        {
            this.mazeNetwork.ClearState();
            await this.RunFirstTimeStep();
            for (int i = 1; i < this.TotalTimeSteps; ++i)
            {
                await this.RunTimeStep();
            }
        }

        /// <summary>
        /// Runs the first time step.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task RunFirstTimeStep()
        {
            this.SimulateForwardPass(this.HiddenStates, this.GetInput(), AppendDirection.VectorLeft);

            var hiddenStateKey = (this.CurrentTileX, this.CurrentTileY);
            if (!this.HiddenStates.ContainsKey(hiddenStateKey))
            {
                this.HiddenStates[hiddenStateKey] = new Tile();
            }

            var hiddenStateTile = this.HiddenStates[(this.CurrentTileX, this.CurrentTileY)];
            hiddenStateTile.Matrix = (Matrix)this.mazeNetwork.HiddenState.ToArray().Last().Clone();

            this.mazeComputationGraphs.Add(this.mazeNetwork.ComputationGraph);

            await this.Backpropagate(this.mazeComputationGraphs.Last());

            this.TimeStep++;
        }

        /// <summary>
        /// Run a time step.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task RunTimeStep()
        {
            var bestExpansion = await this.FindBestExpansion();

            this.FillPlaceholders(this.Grid, this.HiddenStates);  // Ensure the grid and hidden states remain rectangular

            await this.Backpropagate(this.mazeComputationGraphs.Last());

            this.TimeStep++;
        }

        private double SimulateForwardPass(Dictionary<(int X, int Y), Tile> hiddenStates, Matrix input, AppendDirection direction)
        {
            this.mazeNetwork.InitializeState(input);
            var prevHiddenState = this.GetPreviousHiddenState(hiddenStates);
            this.mazeNetwork.AutomaticForwardPropagate(input, prevHiddenState);

            var output = this.mazeNetwork.Output;

            SquaredArclengthEuclideanLossOperation lossOp = SquaredArclengthEuclideanLossOperation.Instantiate(this.mazeNetwork);
            var target = this.GetTarget();
            var loss = lossOp.Forward(output, target);

            // Console.WriteLine($"Loss: {loss[0, 0]}, Direction: {direction.ToString()}, Actual Angle: {lossOp.ActualAngle}");
            var gradient = lossOp.Backward();
            this.LastGradient = gradient;

            this.LastActualAngle = lossOp.ActualAngle;

            return loss[0, 0];
        }

        private async Task Backpropagate(MazeComputationGraph computationGraph)
        {
            Matrix gradient = this.LastGradient;
            Maze maze = this.LastMaze;
            await this.mazeNetwork.AutomaticBackwardPropagate(gradient, computationGraph);

            if (maze != null)
            {
                maze.UpdateModelLayers();
                return;
            }

            this.mazeNetwork.UpdateModelLayers();
        }

        // Assuming this is part of the network operations
        private async Task<((int X, int T) Position, AppendDirection Direction)?> FindBestExpansion()
        {
            double minLoss = double.MaxValue;
            ((int X, int Y), AppendDirection)? bestExpansion = null;
            MazeComputationGraph? bestComputationGraph = null;
            Dictionary<(int X, int Y), Tile>? bestGrid = null;
            Matrix? bestHiddenState = null;
            Matrix? bestGradient = null;
            double? bestActualAngle = null;
            Maze? bestMaze = null;
            (int X, int Y)? bestNextTile = null;

            (int X, int Y) currentTile = (this.CurrentTileX, this.CurrentTileY);

            foreach (AppendDirection direction in Enum.GetValues(typeof(AppendDirection)))
            {
                if (this.IsValidExpansion(currentTile, direction))
                {
                    var simulatedGrid = new Dictionary<(int, int), Tile>(this.Grid);
                    var simulatedHiddenStates = new Dictionary<(int, int), Tile>(this.HiddenStates);
                    bool couldAdd = this.AddTile(simulatedGrid, simulatedHiddenStates, currentTile, direction);
                    if (!couldAdd)
                    {
                        continue;
                    }

                    this.FillPlaceholders(simulatedGrid, simulatedHiddenStates);

                    try
                    {
                        await this.ReinitializeNetwork(simulatedGrid);
                    }
                    catch (Exception)
                    {
                        continue;
                    }

                    Matrix input = this.ConcatenateInput(simulatedGrid);
                    double loss = this.SimulateForwardPass(simulatedHiddenStates, input, direction);

                    this.moreGraphs.Add(this.mazeNetwork.ComputationGraph);

                    if (loss < minLoss)
                    {
                        bestNextTile = this.GetNewPosition(currentTile, direction);
                        minLoss = loss;
                        bestExpansion = (currentTile, direction);
                        bestComputationGraph = this.mazeNetwork.ComputationGraph;
                        bestGradient = this.LastGradient;
                        bestActualAngle = this.LastActualAngle;
                        bestGrid = simulatedGrid;
                        bestHiddenState = (Matrix)this.mazeNetwork.HiddenState.ToArray().Last().Clone();
                        bestMaze = this.mazeNetwork.CloneMaze();
                    }
                }
            }

            if (bestComputationGraph != null && bestGrid != null && bestHiddenState != null && bestNextTile != null && bestGradient != null && bestMaze != null && bestActualAngle != null)
            {
                Console.WriteLine($"Best expansion: {bestExpansion}, Loss: {minLoss}, Actual Angle: {bestActualAngle}");
                this.CurrentTileX = bestNextTile.Value.X;
                this.CurrentTileY = bestNextTile.Value.Y;
                this.mazeComputationGraphs.Add(bestComputationGraph);
                this.Grid = bestGrid;
                this.UpdateMaxDimensions();
                this.UpdateHiddenStateTiles(bestHiddenState);
                this.LastGradient = (Matrix)bestGradient.Clone();
                this.LastMaze = bestMaze;
            }

            return bestExpansion;
        }

        private async Task ReinitializeNetwork(Dictionary<(int X, int Y), Tile> simulatedGrid)
        {
            int[,] structure = new int[this.GridSize, this.GridSize];
            foreach (var pair in simulatedGrid)
            {
                var key = pair.Key;
                var value = pair.Value;
                structure[key.X, key.Y] = value.IsPlaceholder ? 2 : 1;
            }

            await this.mazeNetwork.Reinitialize(structure);
        }
    }
}
