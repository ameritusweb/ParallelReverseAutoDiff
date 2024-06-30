//------------------------------------------------------------------------------
// <copyright file="PradDiagram.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Text;

    /// <summary>
    /// A diagram of a PRAD graph.
    /// </summary>
    public class PradDiagram
    {
        private readonly char[,] grid;
        private readonly int width;
        private readonly int height;

        /// <summary>
        /// Initializes a new instance of the <see cref="PradDiagram"/> class.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public PradDiagram(int width, int height)
        {
            this.width = width;
            this.height = height;
            this.grid = new char[height, width];
            this.InitializeGrid();
        }

        /// <summary>
        /// Add a box to the diagram.
        /// </summary>
        /// <param name="x">The X coordinate.</param>
        /// <param name="y">The Y coordinate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="label">The label.</param>
        public void AddBox(int x, int y, int width, int height, string label)
        {
            // Draw the box
            for (int i = x; i < x + width; i++)
            {
                this.grid[y, i] = '_';
                this.grid[y + height - 1, i] = '_';
            }

            for (int i = y; i < y + height; i++)
            {
                this.grid[i, x] = '|';
                this.grid[i, x + width - 1] = '|';
            }

            // Add the label
            int labelX = x + ((width - label.Length) / 2);
            int labelY = y + (height / 2);
            for (int i = 0; i < label.Length; i++)
            {
                this.grid[labelY, labelX + i] = label[i];
            }
        }

        /// <summary>
        /// Add a line to the diagram.
        /// </summary>
        /// <param name="x1">The X1.</param>
        /// <param name="y1">The Y1.</param>
        /// <param name="x2">The X2.</param>
        /// <param name="y2">The Y2.</param>
        public void AddLine(int x1, int y1, int x2, int y2)
        {
            // Determine the direction of the line
            int dx = x2 - x1;
            int dy = y2 - y1;

            // Draw horizontal line
            for (int x = Math.Min(x1, x2); x <= Math.Max(x1, x2); x++)
            {
                this.SetChar(x, y1, '-');
            }

            // Draw vertical line
            for (int y = Math.Min(y1, y2); y <= Math.Max(y1, y2); y++)
            {
                this.SetChar(x2, y, '|');
            }

            // Add corner if necessary
            if (dx != 0 && dy != 0)
            {
                this.SetChar(x2, y1, '+');
            }
        }

        /// <summary>
        /// To string.
        /// </summary>
        /// <returns>A string.</returns>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int y = 0; y < this.height; y++)
            {
                for (int x = 0; x < this.width; x++)
                {
                    sb.Append(this.grid[y, x]);
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }

        private void InitializeGrid()
        {
            for (int y = 0; y < this.height; y++)
            {
                for (int x = 0; x < this.width; x++)
                {
                    this.grid[y, x] = ' ';
                }
            }
        }

        private void SetChar(int x, int y, char c)
        {
            if (x >= 0 && x < this.width && y >= 0 && y < this.height)
            {
                if (this.grid[y, x] == ' ' || this.grid[y, x] == c)
                {
                    this.grid[y, x] = c;
                }
                else if (this.grid[y, x] != c)
                {
                    this.grid[y, x] = '+';
                }
            }
        }
    }
}
