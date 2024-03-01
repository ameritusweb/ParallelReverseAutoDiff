namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A pattern dimmer.
    /// </summary>
    public class PatternDimmer : PatternGenerator
    {
        private PatternGenerator basePattern;
        private double dimFactor;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternDimmer"/> class.
        /// </summary>
        /// <param name="basePattern">The base pattern.</param>
        /// <param name="dimFactor">The dim factor.</param>
        public PatternDimmer(PatternGenerator basePattern, double dimFactor)
        {
            this.basePattern = basePattern;
            this.dimFactor = Math.Clamp(dimFactor, 0.0, 1.0); // Ensure dimFactor is between 0 and 1
        }

        /// <inheritdoc />
        public override void Generate()
        {
            this.basePattern.Generate(); // Generate the base pattern

            // Apply the dimming effect
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    byte originalShade = this.basePattern[x, y];
                    byte dimmedShade = (byte)(originalShade * this.dimFactor);
                    this.SetCell(x, y, dimmedShade);
                }
            }
        }
    }
}
