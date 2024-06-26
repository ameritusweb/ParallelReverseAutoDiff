//------------------------------------------------------------------------------
// <copyright file="PradSplitResult.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// The result of the computation.
    /// </summary>
    public class PradSplitResult : PradResultBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradSplitResult"/> class.
        /// </summary>
        /// <param name="results">The tensor results.</param>
        /// <param name="gradients">The gradient tensors.</param>
        public PradSplitResult(Tensor[] results, Tensor[] gradients)
        {
            this.Results = results;
            this.Gradients = gradients;
        }

        /// <summary>
        /// Gets the result of the computation.
        /// </summary>
        public Tensor[] Results { get; }
    }
}
