//------------------------------------------------------------------------------
// <copyright file="IClipper.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// Clips gradients.
    /// </summary>
    public interface IClipper
    {
        /// <summary>
        /// Clips the gradients.
        /// </summary>
        /// <param name="tensor">The tensor to clip.</param>
        /// <returns>The clipped tensor.</returns>
        Tensor ClipGradients(Tensor tensor);
    }
}
