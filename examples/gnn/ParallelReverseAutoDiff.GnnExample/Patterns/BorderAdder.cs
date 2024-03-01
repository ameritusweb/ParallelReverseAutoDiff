namespace ParallelReverseAutoDiff.GnnExample.Patterns
{
    /// <summary>
    /// A border adder.
    /// </summary>
    public class BorderAdder : PatternGenerator
    {
        private PatternGenerator basePattern;
        private BorderStyle borderStyle;
        private int borderWidth;
        private byte borderShade;

        /// <summary>
        /// Initializes a new instance of the <see cref="BorderAdder"/> class.
        /// </summary>
        /// <param name="basePattern">The base pattern.</param>
        /// <param name="borderStyle">The border style.</param>
        /// <param name="borderWidth">The border width.</param>
        /// <param name="borderShade">The border shade.</param>
        public BorderAdder(PatternGenerator basePattern, BorderStyle borderStyle, int borderWidth, byte borderShade)
        {
            this.basePattern = basePattern;
            this.borderStyle = borderStyle;
            this.borderWidth = borderWidth;
            this.borderShade = borderShade;
        }

        /// <inheritdoc />
        public override void Generate()
        {
            this.basePattern.Generate(); // Generate the base pattern first

            // Apply the border based on the specified style
            switch (this.borderStyle)
            {
                case BorderStyle.Solid:
                    this.ApplySolidBorder();
                    break;
                case BorderStyle.Dotted:
                    this.ApplyDottedBorder();
                    break;
                case BorderStyle.Dashed:
                    this.ApplyDashedBorder();
                    break;
            }
        }

        private void ApplySolidBorder()
        {
            for (int x = 0; x < 8; x++)
            {
                for (int y = 0; y < 8; y++)
                {
                    if (x < this.borderWidth || x >= 8 - this.borderWidth || y < this.borderWidth || y >= 8 - this.borderWidth)
                    {
                        this.SetCell(x, y, this.borderShade);
                    }
                    else
                    {
                        // Copy the base pattern inside the border
                        this[x, y] = this.basePattern[x, y];
                    }
                }
            }
        }

        private void ApplyDottedBorder()
        {
            for (int i = 0; i < 8; i += 2) // Change the step to control the spacing
            {
                for (int j = 0; j < this.borderWidth; j++)
                {
                    this.SetCell(i, j, this.borderShade); // Top border
                    this.SetCell(i, 7 - j, this.borderShade); // Bottom border
                    this.SetCell(j, i, this.borderShade); // Left border
                    this.SetCell(7 - j, i, this.borderShade); // Right border
                }
            }
        }

        private void ApplyDashedBorder()
        {
            for (int i = 0; i < 8; i += 3) // Change the step to control the dash length
            {
                for (int j = 0; j < this.borderWidth; j++)
                {
                    if (i + 1 < 8)
                    {
                        this.SetCell(i, j, this.borderShade); // Top border
                        this.SetCell(i, 7 - j, this.borderShade); // Bottom border
                        this.SetCell(j, i, this.borderShade); // Left border
                        this.SetCell(7 - j, i, this.borderShade); // Right border

                        this.SetCell(i + 1, j, this.borderShade); // Top border dash continuation
                        this.SetCell(i + 1, 7 - j, this.borderShade); // Bottom border dash continuation
                        this.SetCell(j, i + 1, this.borderShade); // Left border dash continuation
                        this.SetCell(7 - j, i + 1, this.borderShade); // Right border dash continuation
                    }
                }
            }
        }
    }
}
