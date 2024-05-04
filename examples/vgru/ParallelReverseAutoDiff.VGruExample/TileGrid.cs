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
        private readonly int TotalTimeSteps;
        private Dictionary<(int, int), Tile> grid = new Dictionary<(int, int), Tile>();
        private Dictionary<(int, int), Tile> hiddenStates = new Dictionary<(int, int), Tile>();
        private List<MazeComputationGraph> mazeComputationGraphs = new List<MazeComputationGraph>();
        private MazeNetwork mazeNetwork;
        private List<Matrix> inputs;
        private List<double> targets;
        private int timeStep = 0;
        private int maxWidth;
        private int maxHeight;
        private int currentTileX;
        private int currentTileY;

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
            this.TotalTimeSteps = inputs.Count;
            this.targets = targets;
            this.InitializeGrid();
            this.UpdateMaxDimensions();
        }

        /// <summary>
        /// Run time steps.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task RunTimeSteps()
        {
            this.RunFirstTimeStep();
            for (int i = 1; i < this.TotalTimeSteps; ++i)
            {
                await this.RunTimeStep();
            }
        }

        /// <summary>
        /// Runs the first time step.
        /// </summary>
        public void RunFirstTimeStep()
        {
            this.SimulateForwardPass(this.hiddenStates, this.GetInput());

            var hiddenStateKey = (this.currentTileX, this.currentTileY);
            if (!this.hiddenStates.ContainsKey(hiddenStateKey))
            {
                this.hiddenStates[hiddenStateKey] = new Tile();
            }

            var hiddenStateTile = this.hiddenStates[(this.currentTileX, this.currentTileY)];
            hiddenStateTile.Matrix = (Matrix)this.mazeNetwork.HiddenState.ToArray().Last().Clone();

            this.mazeComputationGraphs.Add(this.mazeNetwork.ComputationGraph);
            this.timeStep++;
        }

        /// <summary>
        /// Run a time step.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task RunTimeStep()
        {
            var bestExpansion = await this.FindBestExpansion();

            this.FillPlaceholders(this.grid, this.hiddenStates);  // Ensure the grid and hidden states remain rectangular
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
            this.currentTileX = center;
            this.currentTileY = center;
        }

        private void UpdateMaxDimensions()
        {
            var maxX = this.grid.Keys.Max(p => p.Item1);
            var maxY = this.grid.Keys.Max(p => p.Item2);
            var minX = this.grid.Keys.Min(p => p.Item1);
            var minY = this.grid.Keys.Min(p => p.Item2);
            this.maxWidth = maxY - minY + 1;
            this.maxHeight = maxX - minX + 1;
        }

        private void FillPlaceholders(Dictionary<(int X, int Y), Tile> grid, Dictionary<(int X, int Y), Tile> hiddenStates)
        {
            var minX = grid.Keys.Min(p => p.X);
            var minY = grid.Keys.Min(p => p.Y);
            for (int i = minX; i < minX + this.maxHeight; i++)
            {
                for (int j = minY; j < minY + this.maxWidth; j++)
                {
                    if (!grid.ContainsKey((i, j)))
                    {
                        grid[(i, j)] = new Tile(isPlaceholder: true);
                        hiddenStates[(i, j)] = new Tile(isPlaceholder: true);
                    }
                }
            }
        }

        // Assuming this is part of the network operations
        private async Task<((int X, int T) Position, AppendDirection Direction)?> FindBestExpansion()
        {
            double minLoss = double.MaxValue;
            ((int X, int Y), AppendDirection)? bestExpansion = null;
            MazeComputationGraph? bestComputationGraph = null;
            Dictionary<(int X, int Y), Tile>? bestGrid = null;
            Matrix? bestHiddenState = null;
            (int X, int Y)? bestNextTile = null;

            (int X, int Y) currentTile = (this.currentTileX, this.currentTileY);

            foreach (AppendDirection direction in Enum.GetValues(typeof(AppendDirection)))
            {
                if (this.IsValidExpansion(currentTile, direction))
                {
                    var simulatedGrid = new Dictionary<(int, int), Tile>(this.grid);
                    var simulatedHiddenStates = new Dictionary<(int, int), Tile>(this.hiddenStates);
                    bool couldAdd = this.AddTile(simulatedGrid, simulatedHiddenStates, currentTile, direction);
                    if (!couldAdd)
                    {
                        continue;
                    }

                    this.FillPlaceholders(simulatedGrid, simulatedHiddenStates);

                    await this.ReinitializeNetwork(simulatedGrid);
                    Matrix input = this.ConcatenateInput(simulatedGrid);
                    double loss = this.SimulateForwardPass(simulatedHiddenStates, input);

                    if (loss < minLoss)
                    {
                        bestNextTile = this.GetNewPosition(currentTile, direction);
                        minLoss = loss;
                        bestExpansion = (currentTile, direction);
                        bestComputationGraph = this.mazeNetwork.ComputationGraph;
                        bestGrid = simulatedGrid;
                        bestHiddenState = (Matrix)this.mazeNetwork.HiddenState.ToArray().Last().Clone();
                    }
                }
            }

            if (bestComputationGraph != null && bestGrid != null && bestHiddenState != null && bestNextTile != null)
            {
                this.currentTileX = bestNextTile.Value.X;
                this.currentTileY = bestNextTile.Value.Y;
                this.mazeComputationGraphs.Add(bestComputationGraph);
                this.grid = bestGrid;
                this.UpdateMaxDimensions();
                this.UpdateHiddenStateTiles(bestHiddenState);
            }

            return bestExpansion;
        }

        private void UpdateHiddenStateTiles(Matrix bestHiddenState)
        {
            var rows = this.maxHeight;
            var cols = this.maxWidth;
            var hiddenStateTileMatrices = bestHiddenState.BreakMatrixIntoTiles(rows, cols);
            int minX = this.grid.Keys.Min(x => x.Item1);
            int minY = this.grid.Keys.Min(x => x.Item2);
            for (int i = minX; i < minX + rows; i++)
            {
                for (int j = minY; j < minY + cols; j++)
                {
                    if (!this.hiddenStates.ContainsKey((i, j)))
                    {
                        this.hiddenStates[(i, j)] = new Tile();
                    }

                    this.hiddenStates[(i, j)].Matrix = hiddenStateTileMatrices[i - minX][j - minY];
                }
            }
        }

        private Matrix ConcatenateInput(Dictionary<(int X, int Y), Tile> grid)
        {
            return this.MergeTilesIntoMatrix(grid);
        }

        private async Task ReinitializeNetwork(Dictionary<(int X, int Y), Tile> simulatedGrid)
        {
            int[,] structure = new int[GridSize, GridSize];
            foreach (var pair in simulatedGrid)
            {
                var key = pair.Key;
                var value = pair.Value;
                structure[key.X, key.Y] = value.IsPlaceholder ? 2 : 1;
            }

            await this.mazeNetwork.Reinitialize(structure);
        }

        private bool AddTile(Dictionary<(int X, int Y), Tile> grid, Dictionary<(int X, int Y), Tile> hiddenStates, (int X, int Y) position, AppendDirection direction)
        {
            var newPos = this.GetNewPosition(position, direction);
            if (!grid.ContainsKey(newPos))
            {
                grid[newPos] = new Tile();
                grid[newPos].Matrix = this.GetInput();

                hiddenStates[newPos] = new Tile(isPlaceholder: true);
                this.currentTileX = newPos.X;
                this.currentTileY = newPos.Y;
                return true;
            }

            return false;
        }

        private double SimulateForwardPass(Dictionary<(int X, int Y), Tile> hiddenStates, Matrix input)
        {
            this.mazeNetwork.InitializeState(input);
            var prevHiddenState = this.GetPreviousHiddenState(hiddenStates);
            this.mazeNetwork.AutomaticForwardPropagate(input, prevHiddenState);

            var output = this.mazeNetwork.Output;

            SquaredArclengthEuclideanLossOperation lossOp = SquaredArclengthEuclideanLossOperation.Instantiate(this.mazeNetwork);
            var target = this.GetTarget();
            var loss = lossOp.Forward(output, target);
            return loss[0, 0];
        }

        private Matrix? GetPreviousHiddenState(Dictionary<(int X, int Y), Tile> simulatedGrid)
        {
            if (simulatedGrid.Keys.Count == 0)
            {
                return null;
            }

            return this.MergeTilesIntoMatrix(simulatedGrid);
        }

        private bool IsValidExpansion((int X, int Y) tile, AppendDirection direction)
        {
            if (direction == AppendDirection.VectorLeft || direction == AppendDirection.VectorRight)
            {
                return false;
            }

            // Placeholder logic to check if expansion in this direction is valid
            return true;
        }

        private (int X, int Y) GetNewPosition((int X, int Y) position, AppendDirection direction)
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

        private Matrix MergeTilesIntoMatrix(Dictionary<(int X, int Y), Tile> hiddenStates)
        {
            // Determine the size of the matrix from the keys in the dictionary
            int minRow = hiddenStates.Keys.Min(k => k.X);
            int maxRow = hiddenStates.Keys.Max(k => k.X);
            int minCol = hiddenStates.Keys.Min(k => k.Y);
            int maxCol = hiddenStates.Keys.Max(k => k.Y);

            // Assume each tile has the same size of matrix
            int tileRows = hiddenStates.First().Value.Matrix.Rows;
            int tileCols = hiddenStates.First().Value.Matrix.Cols;

            int totalRows = (maxRow - minRow + 1) * tileRows;
            int totalCols = (maxCol - minCol + 1) * tileCols;

            Matrix resultMatrix = new Matrix(totalRows, totalCols);

            // Fill the result matrix with the matrices from the tiles
            foreach (var key in hiddenStates.Keys)
            {
                Tile tile = hiddenStates[key];
                int startRow = (key.X - minRow) * tileRows;
                int startCol = (key.Y - minCol) * tileCols;

                if (tile.IsPlaceholder)
                {
                    // Fill this tile's area with zeros
                    for (int i = 0; i < tileRows; i++)
                    {
                        for (int j = 0; j < tileCols; j++)
                        {
                            resultMatrix[startRow + i, startCol + j] = 0;
                        }
                    }
                }
                else
                {
                    // Copy this tile's matrix into the appropriate location in the result matrix
                    for (int i = 0; i < tileRows; i++)
                    {
                        for (int j = 0; j < tileCols; j++)
                        {
                            resultMatrix[startRow + i, startCol + j] = tile.Matrix[i, j];
                        }
                    }
                }
            }

            return resultMatrix;
        }
    }
}
