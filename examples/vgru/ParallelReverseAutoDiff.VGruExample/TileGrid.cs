//------------------------------------------------------------------------------
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
    public class TileGrid
    {
        private const int GridSize = 7;
        private Dictionary<(int, int), Tile> grid = new Dictionary<(int, int), Tile>();
        private Dictionary<(int, int), Tile> hiddenStates = new Dictionary<(int, int), Tile>();
        private List<MazeComputationGraph> mazeComputationGraphs = new List<MazeComputationGraph>();
        private MazeNetwork mazeNetwork;
        private List<Matrix> inputs;
        private List<double> targets;
        private int timeStep = 0;
        private int maxWidth;
        private int maxHeight;

        /// <summary>
        /// Initializes a new instance of the <see cref="TileGrid"/> class.
        /// </summary>
        /// <param name="mazeNetwork">The maze network.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="targets">The targets.</param>
        public TileGrid(MazeNetwork mazeNetwork, List<Matrix> inputs, List<double> targets)
        {
            this.mazeNetwork = mazeNetwork;
            this.inputs = inputs;
            this.targets = targets;
            this.InitializeGrid();
            this.InitializeHiddenStates();
            this.UpdateMaxDimensions();
        }

        /// <summary>
        /// Run a time step.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task RunTimeStep()
        {
            var bestExpansion = this.FindBestExpansion();
            if (bestExpansion.HasValue)
            {
                this.ApplyExpansion(bestExpansion.Value.Position, bestExpansion.Value.Direction);
                this.UpdateHiddenStatesForExpansion(bestExpansion.Value.Position, bestExpansion.Value.Direction);
            }

            this.UpdateMaxDimensions();
            this.FillPlaceholders();  // Ensure the grid and hidden states remain rectangular
            await this.Backpropagate();
            this.timeStep++;
        }

        private async Task Backpropagate()
        {
            Matrix gradient = new Matrix();
            await this.mazeNetwork.AutomaticBackwardPropagate(gradient);
        }

        private Matrix GetInput()
        {
            return this.inputs[this.timeStep];
        }

        private double GetTarget()
        {
            return this.targets[this.timeStep];
        }

        private void InitializeGrid()
        {
            var input = this.GetInput();
            var center = (GridSize - 1) / 2;
            this.grid[(center, center)] = new Tile(matrix: input);  // Assuming Tile holds vector and state information
        }

        private void InitializeHiddenStates()
        {
            // Initialize hidden states with the same dimensions as the initial grid
            foreach (var key in this.grid.Keys)
            {
                this.hiddenStates[key] = new Tile();  // Initialize hidden state
            }
        }

        private void UpdateMaxDimensions()
        {
            this.maxWidth = this.grid.Keys.Max(p => p.Item1) + 1;
            this.maxHeight = this.grid.Keys.Max(p => p.Item2) + 1;
        }

        private void UpdateHiddenStatesForExpansion((int X, int Y) position, AppendDirection direction)
        {
            // Add new tiles to hidden states where the grid has expanded
            this.ExpandHiddenStates(position, direction);
        }

        private void ExpandHiddenStates((int X, int Y) position, AppendDirection direction)
        {
            var newPosition = this.GetNewPosition(position, direction);
            if (!this.hiddenStates.ContainsKey(newPosition))
            {
                // Initialize new hidden state tiles as placeholders
                this.hiddenStates[newPosition] = new Tile(isPlaceholder: true);
            }
        }

        private void FillPlaceholders()
        {
            for (int i = 0; i < this.maxHeight; i++)
            {
                for (int j = 0; j < this.maxWidth; j++)
                {
                    if (!this.grid.ContainsKey((i, j)))
                    {
                        this.grid[(i, j)] = new Tile(isPlaceholder: true);
                        this.hiddenStates[(i, j)] = new Tile(isPlaceholder: true);
                    }
                }
            }
        }

        // Assuming this is part of the network operations
        private ((int X, int T) Position, AppendDirection Direction)? FindBestExpansion()
        {
            double minLoss = double.MaxValue;
            ((int X, int Y), AppendDirection)? bestExpansion = null;

            foreach (var tile in this.GetPerimeterTiles())
            {
                foreach (AppendDirection direction in Enum.GetValues(typeof(AppendDirection)))
                {
                    if (this.IsValidExpansion(tile, direction))
                    {
                        var simulatedGrid = new Dictionary<(int, int), Tile>(this.grid);
                        this.AddTile(simulatedGrid, tile, direction);
                        double loss = this.SimulateForwardPass(simulatedGrid);

                        if (loss < minLoss)
                        {
                            minLoss = loss;
                            bestExpansion = (tile, direction);
                        }
                    }
                }
            }

            return bestExpansion;
        }

        private void AddTile(Dictionary<(int X, int Y), Tile> grid, (int X, int Y) position, AppendDirection direction)
        {
            var newPos = this.GetNewPosition(position, direction);
            if (!grid.ContainsKey(newPos))
            {
                grid[newPos] = new Tile();
            }
        }

        private double SimulateForwardPass(Dictionary<(int X, int Y), Tile> simulatedGrid)
        {
            var input = this.GetInput();
            this.mazeNetwork.InitializeState();
            this.mazeNetwork.AutomaticForwardPropagate(input, null);
            this.mazeComputationGraphs.Add(this.mazeNetwork.ComputationGraph);

            var output = this.mazeNetwork.Output;

            SquaredArclengthEuclideanLossOperation lossOp = SquaredArclengthEuclideanLossOperation.Instantiate(this.mazeNetwork);
            var target = this.GetTarget();
            var loss = lossOp.Forward(output, target);
            return loss[0, 0];
        }

        private bool IsValidExpansion((int X, int Y) tile, AppendDirection direction)
        {
            // Placeholder logic to check if expansion in this direction is valid
            return true;
        }

        private List<(int X, int Y)> GetPerimeterTiles()
        {
            return this.grid.Keys.Where(key => this.IsPerimeterTile(key)).ToList();
        }

        private bool IsPerimeterTile((int X, int Y) position)
        {
            foreach (AppendDirection direction in Enum.GetValues(typeof(AppendDirection)))
            {
                var newPos = this.GetNewPosition(position, direction);
                if (!this.grid.ContainsKey(newPos))
                {
                    return true;
                }
            }

            return false;
        }

        private void ApplyExpansion((int X, int T) position, AppendDirection direction)
        {
            // Apply expansion logic to the grid
            var newPos = this.GetNewPosition(position, direction);
            this.grid[newPos] = new Tile();  // Add new tile to the grid
        }

        private (int X, int T) GetNewPosition((int X, int Y) position, AppendDirection direction)
        {
            return direction switch
            {
                AppendDirection.Up => (position.X - 1, position.Y),
                AppendDirection.Down => (position.X + 1, position.Y),
                AppendDirection.Left => (position.X, position.Y - 1),
                AppendDirection.Right => (position.X, position.Y + 1),
                _ => position,
            };
        }
    }
}
