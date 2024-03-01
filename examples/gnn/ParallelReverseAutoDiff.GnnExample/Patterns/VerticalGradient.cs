namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A vertical gradient.
    /// </summary>
    public class VerticalGradient : PatternGenerator
    {
        /// <inheritdoc />
        public override void Generate()
        {
            for (int x = 0; x < 8; x++)
            {
                for (int y = 0; y < 8; y++)
                {
                    byte shade = (byte)(y * (255 / 7));
                    this.SetCell(x, y, shade);
                }
            }
        }
    }
}
