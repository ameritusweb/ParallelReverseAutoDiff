//------------------------------------------------------------------------------
// <copyright file="Layer.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// Represents a layer of a neural network.
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Gets or sets the operations in the layer.
        /// </summary>
        public List<OperationInfo> Operations { get; set; }
    }
}
