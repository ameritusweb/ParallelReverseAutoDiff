//------------------------------------------------------------------------------
// <copyright file="BatchOperation.cs" author="ameritusweb" date="6/24/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <inheritdoc />
    public abstract class BatchOperation : OperationBase, IBatchOperation
    {
        /// <summary>
        /// Abstract method to perform backward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">The upstream gradient.</param>
        /// <returns>The gradients to send to the adjacent backward operations.</returns>
        public abstract BackwardResult[] Backward(DeepMatrix dOutput);
    }
}
