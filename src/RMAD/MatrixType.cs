//------------------------------------------------------------------------------
// <copyright file="MatrixType.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Determines the matrix type.
    /// </summary>
    public enum MatrixType
    {
        /// <summary>
        /// A weight.
        /// </summary>
        Weight,

        /// <summary>
        /// A bias.
        /// </summary>
        Bias,

        /// <summary>
        /// A gradient.
        /// </summary>
        Gradient,

        /// <summary>
        /// An intermediate matrix.
        /// </summary>
        Intermediate,

        /// <summary>
        /// A dynamic.
        /// </summary>
        Dynamic,
    }
}
