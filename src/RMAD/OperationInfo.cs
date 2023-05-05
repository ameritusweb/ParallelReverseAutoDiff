//------------------------------------------------------------------------------
// <copyright file="OperationInfo.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Information about an operation of the computation graph.
    /// </summary>
    public class OperationInfo
    {
        /// <summary>
        /// Gets or sets the ID of the operation.
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// Gets or sets a description of the operation.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Gets or sets the type of the operation.
        /// </summary>
        public string Type { get; set; }

        /// <summary>
        /// Gets or sets the inputs to the operation.
        /// </summary>
        public string[] Inputs { get; set; }

        /// <summary>
        /// Gets or sets where to set the result of the operation.
        /// </summary>
        public string SetResultTo { get; set; }

        /// <summary>
        /// Gets or sets where to place the gradient result.
        /// </summary>
        public string[] GradientResultTo { get; set; }
    }
}
