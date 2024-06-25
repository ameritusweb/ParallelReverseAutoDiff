//------------------------------------------------------------------------------
// <copyright file="PradResult.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The result of the computation.
    /// </summary>
    public class PradResult
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradResult"/> class.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="result">The tensor result.</param>
        /// <param name="gradients">The gradient tensors.</param>
        public PradResult(PradOp operation, Tensor result, Tensor[] gradients)
        {
            this.PradOp = operation;
            this.Result = result;
            this.Gradients = gradients;
        }

        /// <summary>
        /// Gets the result of the computation.
        /// </summary>
        public Tensor Result { get; }

        /// <summary>
        /// Gets the gradient of the input.
        /// </summary>
        public Tensor[] Gradients { get; }

        /// <summary>
        /// Gets the operation.
        /// </summary>
        public PradOp PradOp { get; internal set; }

        /// <summary>
        /// Gets or sets the branches.
        /// </summary>
        public List<PradOp> Branches { get; set; } = new List<PradOp>();
    }
}
