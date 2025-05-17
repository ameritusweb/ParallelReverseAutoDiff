//------------------------------------------------------------------------------
// <copyright file="FourLayersTimeStep.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// Represents a time step with four layers for an operation graph.
    /// </summary>
    public class FourLayersTimeStep
    {
        /// <summary>
        /// Gets or sets the start operations for the time step.
        /// </summary>
        public List<OperationInfo> StartOperations { get; set; }

        /// <summary>
        /// Gets or sets the first layers for the time step.
        /// </summary>
        public List<ILayer> FirstLayers { get; set; }

        /// <summary>
        /// Gets or sets the post first operations for the time step.
        /// </summary>
        public List<OperationInfo> PostFirstOperations { get; set; }

        /// <summary>
        /// Gets or sets the second layers for the time step.
        /// </summary>
        public List<ILayer> SecondLayers { get; set; }

        /// <summary>
        /// Gets or sets the post second operations for the time step.
        /// </summary>
        public List<OperationInfo> PostSecondOperations { get; set; }

        /// <summary>
        /// Gets or sets the third layers for the time step.
        /// </summary>
        public List<ILayer> ThirdLayers { get; set; }

        /// <summary>
        /// Gets or sets the post third operations for the time step.
        /// </summary>
        public List<OperationInfo> PostThirdOperations { get; set; }

        /// <summary>
        /// Gets or sets the fourth layers for the time step.
        /// </summary>
        public List<ILayer> FourthLayers { get; set; }

        /// <summary>
        /// Gets or sets the end operations for the time step.
        /// </summary>
        public List<OperationInfo> EndOperations { get; set; }
    }
}
