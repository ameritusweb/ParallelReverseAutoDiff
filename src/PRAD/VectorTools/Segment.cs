//------------------------------------------------------------------------------
// <copyright file="Segment.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;

    /// <summary>
    /// Represents a segment of a shape defined by angular bounds and inner/outer radius configurations.
    /// Used in shape tensor generation to define portions of shapes.
    /// </summary>
    public class Segment
    {
        /// <summary>
        /// Gets or sets the starting angle of the segment in radians.
        /// </summary>
        public float StartAngle { get; set; }

        /// <summary>
        /// Gets or sets the ending angle of the segment in radians.
        /// </summary>
        public float EndAngle { get; set; }

        /// <summary>
        /// Gets or sets the configuration for the outer radius of the segment.
        /// </summary>
        public RadiusConfig OuterRadius { get; set; }

        /// <summary>
        /// Gets or sets the configuration for the inner radius of the segment.
        /// </summary>
        public RadiusConfig InnerRadius { get; set; }

        /// <summary>
        /// Validates the segment configuration to ensure it contains valid values
        /// and maintains proper relationships between inner and outer radii.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// Thrown when start angle is greater than or equal to end angle,
        /// when either radius configuration is missing, or when inner radius
        /// could potentially exceed outer radius.
        /// </exception>
        public void Validate()
        {
            if (this.StartAngle >= this.EndAngle)
            {
                throw new ArgumentException("StartAngle must be less than EndAngle");
            }

            this.OuterRadius?.Validate();
            this.InnerRadius?.Validate();

            if (this.OuterRadius == null || this.InnerRadius == null)
            {
                throw new ArgumentException("Both OuterRadius and InnerRadius must be specified");
            }

            // Check that inner radius is always less than outer radius, even with variations
            float maxInnerRadius = this.InnerRadius.Base + (this.InnerRadius.Variation?.Amplitude ?? 0);
            float minOuterRadius = this.OuterRadius.Base - (this.OuterRadius.Variation?.Amplitude ?? 0);

            if (maxInnerRadius >= minOuterRadius)
            {
                throw new ArgumentException("InnerRadius must always be less than OuterRadius");
            }
        }
    }
}