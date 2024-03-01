namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A horizontal gradient.
    /// </summary>
    public class HorizontalGradient : PatternGenerator
    {
        /// <inheritdoc />
        public override void Generate()
        {
            for (int x = 0; x < 8; x++)
            {
                byte shade = (byte)(x * (255 / 7));
                for (int y = 0; y < 8; y++)
                {
                    this.SetCell(x, y, shade);
                }
            }
        }
    }
}
