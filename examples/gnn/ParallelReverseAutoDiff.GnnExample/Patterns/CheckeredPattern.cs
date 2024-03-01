namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A checkered pattern.
    /// </summary>
    public class CheckeredPattern : PatternGenerator
    {
        /// <inheritdoc />
        public override void Generate()
        {
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    byte shade = (x + y) % 2 == 0 ? (byte)0 : (byte)255;
                    this.SetCell(x, y, shade);
                }
            }
        }
    }
}
