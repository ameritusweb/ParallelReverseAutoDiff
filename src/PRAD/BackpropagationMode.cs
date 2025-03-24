//------------------------------------------------------------------------------
// <copyright file="BackpropagationMode.cs" author="ameritusweb" date="3/24/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// The backpropagation mode.
    /// </summary>
    public enum BackpropagationMode
    {
        /// <summary>
        /// Replace the seed gradient.
        /// </summary>
        Replace,

        /// <summary>
        /// Accumulate and track gradients.
        /// </summary>
        Accumulate,
    }
}
