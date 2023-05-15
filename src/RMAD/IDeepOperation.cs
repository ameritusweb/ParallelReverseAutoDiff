//------------------------------------------------------------------------------
// <copyright file="IDeepOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Define the abstract base class for all deep operations.
    /// </summary>
    public interface IDeepOperation : IOperationBase
    {
        /// <summary>
        /// Abstract method to perform backward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">The upstream gradient.</param>
        /// <returns>The gradients to send to the adjacent backward operations.</returns>
        BackwardResult Backward(DeepMatrix dOutput);
    }
}
