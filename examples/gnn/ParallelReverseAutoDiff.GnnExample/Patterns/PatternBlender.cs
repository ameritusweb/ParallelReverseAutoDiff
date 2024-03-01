namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A pattern blender.
    /// </summary>
    public class PatternBlender : PatternGenerator
    {
        private readonly PatternGenerator pattern1;
        private readonly PatternGenerator pattern2;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternBlender"/> class.
        /// </summary>
        /// <param name="pattern1">The first pattern.</param>
        /// <param name="pattern2">The second pattern.</param>
        public PatternBlender(PatternGenerator pattern1, PatternGenerator pattern2)
        {
            this.pattern1 = pattern1;
            this.pattern2 = pattern2;
        }

        /// <inheritdoc />
        public override void Generate()
        {
            // Ensure both patterns are generated.
            this.pattern1.Generate();
            this.pattern2.Generate();

            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    // Compute the simple average of the two patterns at each cell.
                    byte averageShade = (byte)((this.pattern1[x, y] + this.pattern2[x, y]) / 2);
                    this.SetCell(x, y, averageShade);
                }
            }
        }
    }
}
