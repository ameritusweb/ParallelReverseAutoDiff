//------------------------------------------------------------------------------
// <copyright file="TimeStep.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// Represents a time step for an operation graph.
    /// </summary>
    public class TimeStep
    {
        /// <summary>
        /// Gets or sets the start operations for the time step.
        /// </summary>
        public List<OperationInfo> StartOperations { get; set; }

        /// <summary>
        /// Gets or sets the layers for the time step.
        /// </summary>
        public List<Layer> Layers { get; set; }

        /// <summary>
        /// Gets or sets the end operations for the time step.
        /// </summary>
        public List<OperationInfo> EndOperations { get; set; }
    }
}
