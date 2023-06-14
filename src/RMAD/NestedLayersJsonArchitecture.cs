//------------------------------------------------------------------------------
// <copyright file="NestedLayersJsonArchitecture.cs" author="ameritusweb" date="5/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The nested layers JSON architecture for a neural network.
    /// </summary>
    public class NestedLayersJsonArchitecture
    {
        /// <summary>
        /// Gets or sets the time steps for a neural network.
        /// </summary>
        public List<NestedLayersTimeStep> TimeSteps { get; set; }
    }
}
