//------------------------------------------------------------------------------
// <copyright file="GridVector.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    /// <summary>
    /// A grid vector.
    /// </summary>
    public class GridVector
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GridVector"/> class.
        /// </summary>
        /// <param name="startPiece">The start piece.</param>
        /// <param name="endPiece">The end piece.</param>
        /// <param name="start">The start.</param>
        /// <param name="end">Tne end.</param>
        /// <param name="angle">The angle.</param>
        public GridVector(string startPiece, string endPiece, string start, string end, double angle)
        {
            this.StartPiece = startPiece;
            this.EndPiece = endPiece;
            this.Start = start;
            this.End = end;
            this.Angle = angle;
        }

        /// <summary>
        /// Gets or sets the start.
        /// </summary>
        public string Start { get; set; }

        /// <summary>
        /// Gets or sets the end.
        /// </summary>
        public string End { get; set; }

        /// <summary>
        /// Gets or sets the start piece.
        /// </summary>
        public string StartPiece { get; set; }

        /// <summary>
        /// Gets or sets the end piece.
        /// </summary>
        public string EndPiece { get; set; }

        /// <summary>
        /// Gets or sets the angle.
        /// </summary>
        public double Angle { get; set; }
    }
}
