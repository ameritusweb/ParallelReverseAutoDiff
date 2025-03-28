//------------------------------------------------------------------------------
// <copyright file="ShapeConfigs.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Contains a collection of shape configurations identified by name.
    /// Used to define multiple shapes that can be referenced by identifier.
    /// </summary>
    public class ShapeConfigs
    {
        /// <summary>
        /// Gets or sets the dictionary of shape configurations keyed by shape name.
        /// </summary>
        public Dictionary<string, ShapeConfig> Shapes { get; set; }

        /// <summary>
        /// Validates all shape configurations to ensure they contain valid values.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// Thrown when the dictionary is null or empty, or when any shape configuration is invalid.
        /// </exception>
        public void Validate()
        {
            if (this.Shapes == null || this.Shapes.Count == 0)
            {
                throw new ArgumentException("At least one shape must be specified");
            }

            foreach (var shape in this.Shapes.Values)
            {
                shape.Validate();
            }
        }
    }
}