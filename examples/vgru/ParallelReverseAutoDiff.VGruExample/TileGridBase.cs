//------------------------------------------------------------------------------
// <copyright file="TileGridBase.cs" author="ameritusweb" date="5/2/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork;

    /// <summary>
    /// The base class for a tile grid.
    /// </summary>
    public abstract class TileGridBase
    {
        private const int GRIDSIZE = 9;
        private readonly int TOTALTIMESTEPS;
        private Dictionary<(int, int), Tile> grid = new Dictionary<(int, int), Tile>();
        private Dictionary<(int, int), Tile> hiddenStates = new Dictionary<(int, int), Tile>();

        private Matrix lastGradient;
        private double lastActualAngle;
        private Maze lastMaze;
        private List<Matrix> inputs;
        private List<double> targets;
        private int timeStep = 0;
        private int maxWidth;
        private int maxHeight;
        private int currentTileX;
        private int currentTileY;

        /// <summary>
        /// Initializes a new instance of the <see cref="TileGridBase"/> class.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        public TileGridBase(List<Matrix> inputs)
        {
            this.TOTALTIMESTEPS = inputs.Count;
            this.InitializeGrid();
            this.UpdateMaxDimensions();
        }

        /// <summary>
        /// Gets the grid size.
        /// </summary>
        protected int GridSize => TileGridBase.GRIDSIZE;

        /// <summary>
        /// Gets the total time steps.
        /// </summary>
        protected int TotalTimeSteps => this.TOTALTIMESTEPS;

        /// <summary>
        /// Gets or sets the grid.
        /// </summary>
        protected Dictionary<(int X, int Y), Tile> Grid { get => this.grid; set => this.grid = value; }

        /// <summary>
        /// Gets or sets the hidden states.
        /// </summary>
        protected Dictionary<(int X, int Y), Tile> HiddenStates { get => this.hiddenStates; set => this.hiddenStates = value; }

        /// <summary>
        /// Gets or sets the last gradient.
        /// </summary>
        protected Matrix LastGradient { get => this.lastGradient; set => this.lastGradient = value; }

        /// <summary>
        /// Gets or sets the last actual angle.
        /// </summary>
        protected double LastActualAngle { get => this.lastActualAngle; set => this.lastActualAngle = value; }

        /// <summary>
        /// Gets or sets the last maze.
        /// </summary>
        protected Maze LastMaze { get => this.lastMaze; set => this.lastMaze = value; }

        /// <summary>
        /// Gets or sets Inputs.
        /// </summary>
        protected List<Matrix> Inputs { get => this.inputs; set => this.inputs = value; }

        /// <summary>
        /// Gets or sets targets.
        /// </summary>
        protected List<double> Targets { get => this.targets; set => this.targets = value; }

        /// <summary>
        /// Gets or sets time step.
        /// </summary>
        protected int TimeStep { get => this.timeStep; set => this.timeStep = value; }

        /// <summary>
        /// Gets or sets max width.
        /// </summary>
        protected int MaxWidth { get => this.maxWidth; set => this.maxWidth = value; }

        /// <summary>
        /// Gets or sets max height.
        /// </summary>
        protected int MaxHeight { get => this.maxHeight; set => this.maxHeight = value; }

        /// <summary>
        /// Gets or sets Current tile X.
        /// </summary>
        protected int CurrentTileX { get => this.currentTileX; set => this.currentTileX = value; }

        /// <summary>
        /// Gets or sets Current tile Y.
        /// </summary>
        protected int CurrentTileY { get => this.currentTileY; set => this.currentTileY = value; }

        /// <summary>
        /// Gets the input.
        /// </summary>
        /// <returns>The input.</returns>
        protected virtual Matrix GetInput()
        {
            return this.inputs[this.timeStep];
        }

        /// <summary>
        /// Gets the target.
        /// </summary>
        /// <returns>The target.</returns>
        protected virtual double GetTarget()
        {
            return this.targets[this.timeStep];
        }

        /// <summary>
        /// Inititialize the grid.
        /// </summary>
        protected virtual void InitializeGrid()
        {
            var input = this.GetInput();
            var center = (GRIDSIZE - 1) / 2;
            this.grid[(center, center)] = new Tile(matrix: input);  // Assuming Tile holds vector and state information
            this.currentTileX = center;
            this.currentTileY = center;
        }

        /// <summary>
        /// Update the max dimensions.
        /// </summary>
        protected virtual void UpdateMaxDimensions()
        {
            var maxX = this.grid.Keys.Max(p => p.Item1);
            var maxY = this.grid.Keys.Max(p => p.Item2);
            var minX = this.grid.Keys.Min(p => p.Item1);
            var minY = this.grid.Keys.Min(p => p.Item2);
            this.maxWidth = maxY - minY + 1;
            this.maxHeight = maxX - minX + 1;
        }

        /// <summary>
        /// Fill the placeholders.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        protected virtual void FillPlaceholders(Dictionary<(int X, int Y), Tile> grid, Dictionary<(int X, int Y), Tile> hiddenStates)
        {
            var minX = grid.Keys.Min(p => p.X);
            var minY = grid.Keys.Min(p => p.Y);
            var maxX = grid.Keys.Max(p => p.X);
            var maxY = grid.Keys.Max(p => p.Y);
            var maxHeight = maxX - minX + 1;
            var maxWidth = maxY - minY + 1;
            for (int i = minX; i < minX + maxHeight; i++)
            {
                for (int j = minY; j < minY + maxWidth; j++)
                {
                    if (!grid.ContainsKey((i, j)))
                    {
                        grid[(i, j)] = new Tile(isPlaceholder: true);
                        hiddenStates[(i, j)] = new Tile(isPlaceholder: true);
                    }
                }
            }
        }

        /// <summary>
        /// Update hidden state tiles.
        /// </summary>
        /// <param name="bestHiddenState">The best hidden state.</param>
        protected virtual void UpdateHiddenStateTiles(Matrix bestHiddenState)
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

        /// <summary>
        /// Concatenates the input.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <returns>A matrix.</returns>
        protected virtual Matrix ConcatenateInput(Dictionary<(int X, int Y), Tile> grid)
        {
            return this.MergeTilesIntoMatrix(grid);
        }

        /// <summary>
        /// Adds a tile.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <param name="position">The position.</param>
        /// <param name="direction">The direction.</param>
        /// <returns>A bool.</returns>
        protected virtual bool AddTile(Dictionary<(int X, int Y), Tile> grid, Dictionary<(int X, int Y), Tile> hiddenStates, (int X, int Y) position, AppendDirection direction)
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
            else if (grid[newPos].IsPlaceholder)
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

        /// <summary>
        /// Gets the previous hidden state.
        /// </summary>
        /// <param name="simulatedGrid">The simulated grid.</param>
        /// <returns>The matrix.</returns>
        protected virtual Matrix? GetPreviousHiddenState(Dictionary<(int X, int Y), Tile> simulatedGrid)
        {
            if (simulatedGrid.Keys.Count == 0)
            {
                return null;
            }

            return this.MergeTilesIntoMatrix(simulatedGrid);
        }

        /// <summary>
        /// Is a valid expansion.
        /// </summary>
        /// <param name="tile">The tile.</param>
        /// <param name="direction">The direction.</param>
        /// <returns>A bool.</returns>
        protected virtual bool IsValidExpansion((int X, int Y) tile, AppendDirection direction)
        {
            if (direction == AppendDirection.VectorLeft || direction == AppendDirection.VectorRight)
            {
                return false;
            }

            // Placeholder logic to check if expansion in this direction is valid
            return true;
        }

        /// <summary>
        /// Gets the new position.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="direction">The direction.</param>
        /// <returns>The new position.</returns>
        protected virtual (int X, int Y) GetNewPosition((int X, int Y) position, AppendDirection direction)
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

        /// <summary>
        /// Merge the tiles into a matrix.
        /// </summary>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <returns>The matrix.</returns>
        protected virtual Matrix MergeTilesIntoMatrix(Dictionary<(int X, int Y), Tile> hiddenStates)
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
