//------------------------------------------------------------------------------
// <copyright file="DualLayersTimeStep.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// Represents a time step with dual layers for an operation graph.
    /// </summary>
    public class DualLayersTimeStep
    {
        /// <summary>
        /// Gets or sets the start operations for the time step.
        /// </summary>
        public List<OperationInfo> StartOperations { get; set; }

        /// <summary>
        /// Gets or sets the first layers for the time step.
        /// </summary>
        public List<Layer> FirstLayers { get; set; }

        /// <summary>
        /// Gets or sets the middle operations for the time step.
        /// </summary>
        public List<OperationInfo> MiddleOperations { get; set; }

        /// <summary>
        /// Gets or sets the second layers for the time step.
        /// </summary>
        public List<Layer> SecondLayers { get; set; }

        /// <summary>
        /// Gets or sets the end operations for the time step.
        /// </summary>
        public List<OperationInfo> EndOperations { get; set; }
    }
}
