//------------------------------------------------------------------------------
// <copyright file="VariationConfig.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;

    /// <summary>
    /// Represents configuration for variations applied to radius values.
    /// Defines frequency and amplitude of sinusoidal variations.
    /// </summary>
    public class VariationConfig
    {
        /// <summary>
        /// Gets or sets the frequency of sinusoidal variation.
        /// Higher values create more oscillations as the angle increases.
        /// </summary>
        public float Frequency { get; set; }

        /// <summary>
        /// Gets or sets the amplitude of sinusoidal variation.
        /// Determines the maximum amount of deviation from the base radius.
        /// </summary>
        public float Amplitude { get; set; }

        /// <summary>
        /// Validates the configuration to ensure it contains valid values.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// Thrown when frequency is zero or negative, or when amplitude is negative.
        /// </exception>
        public void Validate()
        {
            if (this.Frequency <= 0)
            {
                throw new ArgumentException("Frequency must be positive");
            }

            if (this.Amplitude < 0)
            {
                throw new ArgumentException("Amplitude cannot be negative");
            }
        }
    }
}