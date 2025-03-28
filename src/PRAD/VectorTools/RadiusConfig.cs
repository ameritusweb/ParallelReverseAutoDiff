//------------------------------------------------------------------------------
// <copyright file="RadiusConfig.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;

    /// <summary>
    /// Represents configuration for radius generation in shape tensor generation.
    /// Defines the base radius and optional variation parameters.
    /// </summary>
    public class RadiusConfig
    {
        /// <summary>
        /// Gets or sets the base radius value, which serves as the central radius value.
        /// </summary>
        public float Base { get; set; }

        /// <summary>
        /// Gets or sets the optional configuration for radius variation around the base value.
        /// When null, the radius will be constant at the base value.
        /// </summary>
        public VariationConfig? Variation { get; set; }

        /// <summary>
        /// Validates the configuration to ensure it contains valid values.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when the base radius is zero or negative.</exception>
        public void Validate()
        {
            if (this.Base <= 0)
            {
                throw new ArgumentException("Base radius must be positive");
            }

            this.Variation?.Validate();
        }
    }
}