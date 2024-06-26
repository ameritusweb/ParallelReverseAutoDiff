//------------------------------------------------------------------------------
// <copyright file="PradResultBase.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The base class for the result of the computation.
    /// </summary>
    public abstract class PradResultBase
    {
        /// <summary>
        /// Gets or sets the gradient of the input.
        /// </summary>
        public Tensor[] Gradients { get; protected set; }

        /// <summary>
        /// Gets the operation.
        /// </summary>
        public PradOp PradOp { get; internal set; }

        /// <summary>
        /// Gets or sets the branches.
        /// </summary>
        public List<PradOp> Branches { get; set; } = new List<PradOp>();

        /// <summary>
        /// Gets or sets the split branches.
        /// </summary>
        public List<PradOp> SplitBranches { get; set; } = new List<PradOp>();
    }
}
