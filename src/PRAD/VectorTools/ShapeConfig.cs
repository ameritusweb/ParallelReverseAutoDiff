//------------------------------------------------------------------------------
// <copyright file="ShapeConfig.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Defines the configuration for a shape to be generated as a tensor.
    /// Contains parameters for the number of points and collection of segments.
    /// </summary>
    public class ShapeConfig
    {
        /// <summary>
        /// Gets or sets the number of points to generate along the entire shape (2π radians).
        /// The actual number of points per segment is proportional to the segment's angular span.
        /// </summary>
        public int NumPoints { get; set; }

        /// <summary>
        /// Gets or sets the collection of segments that define the shape.
        /// </summary>
        public List<Segment> Segments { get; set; }

        /// <summary>
        /// Validates the shape configuration to ensure it contains valid values.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// Thrown when NumPoints is not positive or when no segments are specified.
        /// </exception>
        public void Validate()
        {
            if (this.NumPoints <= 0)
            {
                throw new ArgumentException("NumPoints must be positive");
            }

            if (this.Segments == null || this.Segments.Count == 0)
            {
                throw new ArgumentException("At least one segment must be specified");
            }

            foreach (var segment in this.Segments)
            {
                segment.Validate();
            }
        }
    }
}