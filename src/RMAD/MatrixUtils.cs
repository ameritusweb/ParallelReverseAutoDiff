//------------------------------------------------------------------------------
// <copyright file="MatrixUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Matrix utilities for reverse mode automatic differentiation.
    /// </summary>
    public static class MatrixUtils
    {
        /// <summary>
        /// Converts the tuple to an array of matrices.
        /// </summary>
        /// <param name="dOutput">The tuple of matrices.</param>
        /// <returns>The array of matrices.</returns>
        public static Matrix?[] Reassemble((Matrix?, Matrix?) dOutput)
        {
            return new[] { dOutput.Item1, dOutput.Item2 };
        }
    }
}
