//------------------------------------------------------------------------------
// <copyright file="InitializationType.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// The type of initialization to use.
    /// </summary>
    public enum InitializationType
    {
        /// <summary>
        /// He initialization.
        /// </summary>
        He,

        /// <summary>
        /// Xavier/Glorot initialization.
        /// </summary>
        Xavier,

        /// <summary>
        /// Initialize with zeroes
        /// </summary>
        Zeroes,

        /// <summary>
        /// He initialization for adjacency matrices.
        /// </summary>
        HeAdjacency,
    }
}
