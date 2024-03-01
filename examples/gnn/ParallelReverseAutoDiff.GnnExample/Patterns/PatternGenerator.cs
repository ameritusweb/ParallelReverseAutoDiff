namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A pattern generator.
    /// </summary>
    public abstract class PatternGenerator
    {
        private byte[,] grid = new byte[8, 8];

        /// <summary>
        /// Gets the shade at the specified coordinates.
        /// </summary>
        /// <param name="x">X.</param>
        /// <param name="y">Y.</param>
        /// <returns>The shade.</returns>
        public byte this[int x, int y]
        {
            get => this.grid[x, y];
            set => this.grid[x, y] = value;
        }

        /// <summary>
        /// Generates a pattern.
        /// </summary>
        public abstract void Generate();

        /// <summary>
        /// Prints a pattern.
        /// </summary>
        public void Print()
        {
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    Console.Write($"{this.grid[x, y],3} ");
                }

                Console.WriteLine();
            }
        }

        /// <summary>
        /// Sets a cell.
        /// </summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        /// <param name="shade">The shade.</param>
        protected void SetCell(int x, int y, byte shade)
        {
            if (x >= 0 && x < 8 && y >= 0 && y < 8)
            {
                this.grid[x, y] = shade;
            }
        }
    }
}
