//------------------------------------------------------------------------------
// <copyright file="LayerInfo.cs" author="ameritusweb" date="5/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Matrix index.
    /// </summary>
    public struct LayerInfo
    {
        /// <summary>
        /// Gets or sets the time step.
        /// </summary>
        public int TimeStep { get; set; }

        /// <summary>
        /// Gets or sets the layer index.
        /// </summary>
        public int Layer { get; set; }

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
