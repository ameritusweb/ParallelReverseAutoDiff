namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A wave pattern.
    /// </summary>
    public class WavePattern : PatternGenerator
    {
        /// <inheritdoc />
        public override void Generate()
        {
            for (int x = 0; x < 8; x++)
            {
                int y = (int)(4 + (Math.Sin(x / 2.0) * 3));
                this.SetCell(x, y, 255);
            }
        }
    }
}
