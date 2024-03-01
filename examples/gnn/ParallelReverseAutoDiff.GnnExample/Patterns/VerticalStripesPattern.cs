namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A vertical stripes pattern.
    /// </summary>
    public class VerticalStripesPattern : PatternGenerator
    {
        private byte shade1;
        private byte shade2;
        private int stripeWidth;

        /// <summary>
        /// Initializes a new instance of the <see cref="VerticalStripesPattern"/> class.
        /// </summary>
        /// <param name="shade1">The first shade.</param>
        /// <param name="shade2">The second shade.</param>
        /// <param name="stripeWidth">The stripe width.</param>
        public VerticalStripesPattern(byte shade1, byte shade2, int stripeWidth)
        {
            this.shade1 = shade1;
            this.shade2 = shade2;
            this.stripeWidth = stripeWidth;
        }

        /// <inheritdoc />
        public override void Generate()
        {
            for (int x = 0; x < 8; x++)
            {
                byte currentShade = (x / this.stripeWidth) % 2 == 0 ? this.shade1 : this.shade2;
                for (int y = 0; y < 8; y++)
                {
                    this.SetCell(x, y, currentShade);
                }
            }
        }
    }
}
