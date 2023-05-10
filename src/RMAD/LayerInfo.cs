//------------------------------------------------------------------------------
// <copyright file="LayerInfo.cs" author="ameritusweb" date="5/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Matrix index.
    /// </summary>
    public struct LayerInfo : IEquatable<LayerInfo>
    {
        /// <summary>
        /// An empty layer info.
        /// </summary>
        public static readonly LayerInfo Empty;

        /// <summary>
        /// Gets or sets the time step.
        /// </summary>
        public int TimeStep { get; set; }

        /// <summary>
        /// Gets or sets the layer index.
        /// </summary>
        public int Layer { get; set; }

        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="LayerInfo"/>.
        /// </summary>
        /// <param name="other">The other object.</param>
        /// <returns>Whether they are equal.</returns>
        public bool Equals(LayerInfo other)
        {
            return this.TimeStep == other.TimeStep && this.Layer == other.Layer;
        }

        /// <summary>
        /// To the specific ID string.
        /// </summary>
        /// <returns>The specific ID string.</returns>
        public override string ToString()
        {
            return "_" + this.TimeStep.ToString() + "_" + this.Layer.ToString();
        }
    }
}
