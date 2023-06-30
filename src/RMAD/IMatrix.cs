//------------------------------------------------------------------------------
// <copyright file="IMatrix.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Abstract base class for all matrices.
    /// </summary>
    public interface IMatrix
    {
        /// <summary>
        /// Gets the number of rows in a matrix, the depth of a deep matrix, or the count of a four-dimensional matrix.
        /// </summary>
        public int Count { get; }
    }
}
