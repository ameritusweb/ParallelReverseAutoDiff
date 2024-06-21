//------------------------------------------------------------------------------
// <copyright file="PradResult.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// The result of the computation.
    /// </summary>
    public class PradResult
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradResult"/> class.
        /// </summary>
        /// <param name="result">The tensor result.</param>
        /// <param name="gradient">The gradient tensor.</param>
        public PradResult(Tensor result, Tensor gradient)
        {
            this.Result = result;
            this.Gradient = gradient;
        }

        /// <summary>
        /// Gets the result of the computation.
        /// </summary>
        public Tensor Result { get; }

        /// <summary>
        /// Gets the gradient of the input.
        /// </summary>
        public Tensor Gradient { get; }
    }
}
