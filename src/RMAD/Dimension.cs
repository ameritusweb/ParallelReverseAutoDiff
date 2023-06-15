//------------------------------------------------------------------------------
// <copyright file="Dimension.cs" author="ameritusweb" date="5/17/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Represents dimension information.
    /// </summary>
    public struct Dimension : IEquatable<Dimension>
    {
        /// <summary>
        /// An empty dimension.
        /// </summary>
        public static readonly Dimension Empty;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dimension"/> struct.
        /// </summary>
        /// <param name="depth">The depth.</param>
        /// <param name="height">The rows.</param>
        /// <param name="width">The columns.</param>
        public Dimension(int depth, int height, int width)
        {
            this.Depth = depth;
            this.Height = height;
            this.Width = width;
        }

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// Gets or sets the depth.
        /// </summary>
        public int Depth { get; set; }

        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="Dimension"/>.
        /// </summary>
        /// <param name="other">The other object.</param>
        /// <returns>Whether they are equal.</returns>
        public bool Equals(Dimension other)
        {
            return this.Height == other.Height && this.Width == other.Width && this.Depth == other.Depth;
        }
    }
}
