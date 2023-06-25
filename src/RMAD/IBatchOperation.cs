//------------------------------------------------------------------------------
// <copyright file="IBatchOperation.cs" author="ameritusweb" date="6/24/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Define the abstract base class for all batch operations.
    /// </summary>
    public interface IBatchOperation : IOperationBase
    {
        /// <summary>
        /// Abstract method to perform backward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">The upstream gradient.</param>
        /// <returns>The gradients to send to the adjacent backward operations.</returns>
        BackwardResult[] Backward(DeepMatrix dOutput);
    }
}
