//------------------------------------------------------------------------------
// <copyright file="JsonArchitecture.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The JSON architecture for a neural network.
    /// </summary>
    public class JsonArchitecture
    {
        /// <summary>
        /// Gets or sets the time steps for a neural network.
        /// </summary>
        public List<TimeStep> TimeSteps { get; set; }
    }
}
