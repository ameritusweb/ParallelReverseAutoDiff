//------------------------------------------------------------------------------
// <copyright file="DualLayersJsonArchitecture.cs" author="ameritusweb" date="5/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The dual layers JSON architecture for a neural network.
    /// </summary>
    public class DualLayersJsonArchitecture
    {
        /// <summary>
        /// Gets or sets the time steps for a neural network.
        /// </summary>
        public List<DualLayersTimeStep> TimeSteps { get; set; }
    }
}
