namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A swirl pattern.
    /// </summary>
    public class SwirlPattern : PatternGenerator
    {
        /// <inheritdoc />
        public override void Generate()
        {
            // Center of the swirl
            double centerX = 3.5;
            double centerY = 3.5;

            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    // Calculate distance from the center
                    double distance = Math.Sqrt(((x - centerX) * (x - centerX)) + ((y - centerY) * (y - centerY)));

                    // Normalize and calculate angle for swirl effect
                    double angle = Math.Atan2(y - centerY, x - centerX) + (distance * Math.PI / 8);

                    // Use sin to create alternating pattern and map distance to shade
                    byte shade = (byte)((Math.Sin(angle) * 127.5) + 127.5);

                    this.SetCell(x, y, shade);
                }
            }
        }
    }
}
