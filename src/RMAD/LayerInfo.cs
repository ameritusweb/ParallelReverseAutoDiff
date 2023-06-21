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
        /// Initializes a new instance of the <see cref="LayerInfo"/> struct.
        /// </summary>
        /// <param name="timeStep">The time step.</param>
        /// <param name="layer">The layer.</param>
        /// <param name="nestedLayer">The nested layer.</param>
        public LayerInfo(int timeStep, int layer, int nestedLayer)
        {
            this.TimeStep = timeStep;
            this.Layer = layer;
            this.NestedLayer = nestedLayer;
            this.Type = LayerInfoType.Nested;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LayerInfo"/> struct.
        /// </summary>
        /// <param name="timeStep">The time step.</param>
        /// <param name="layer">The layer.</param>
        public LayerInfo(int timeStep, int layer)
        {
            this.TimeStep = timeStep;
            this.Layer = layer;
            this.NestedLayer = default(int);
            this.Type = LayerInfoType.Normal;
        }

        /// <summary>
        /// Gets or sets the time step.
        /// </summary>
        public int TimeStep { get; set; }

        /// <summary>
        /// Gets or sets the layer index.
        /// </summary>
        public int Layer { get; set; }

        /// <summary>
        /// Gets or sets the nested layer index.
        /// </summary>
        public int NestedLayer { get; set; }

        /// <summary>
        /// Gets or sets the layer info type.
        /// </summary>
        public LayerInfoType Type { get; set; }

        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="LayerInfo"/>.
        /// </summary>
        /// <param name="other">The other object.</param>
        /// <returns>Whether they are equal.</returns>
        public bool Equals(LayerInfo other)
        {
            return this.TimeStep == other.TimeStep && this.Layer == other.Layer && this.NestedLayer == other.NestedLayer;
        }

        /// <summary>
        /// To the specific ID string.
        /// </summary>
        /// <returns>The specific ID string.</returns>
        public override string ToString()
        {
            return "_" + this.TimeStep.ToString() + "_" + this.Layer.ToString();
        }

        /// <summary>
        /// To the nested specific ID string.
        /// </summary>
        /// <returns>The nested specific ID string.</returns>
        public string ToNestedString()
        {
            return "_" + this.TimeStep.ToString() + "_" + this.Layer.ToString() + "_" + this.NestedLayer.ToString();
        }
    }
}
