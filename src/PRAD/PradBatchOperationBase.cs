//------------------------------------------------------------------------------
// <copyright file="PradBatchOperationBase.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A base class for PRAD operations.
    /// </summary>
    /// <typeparam name="TOperation">The operation type.</typeparam>
    /// <typeparam name="TSubType">The subtype.</typeparam>
    /// <typeparam name="TReturnType">The return type.</typeparam>"
    public abstract class PradBatchOperationBase<TOperation, TSubType, TReturnType> : OperationBase, IBatchOperation
        where TOperation : PradBatchOperationBase<TOperation, TSubType, TReturnType>
    {
        /// <summary>
        /// Abstract method to perform forward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="input">The input to the forward pass.</param>
        /// <returns>The result of the forward pass.</returns>
        public abstract TReturnType Forward(TSubType input);

        /// <summary>
        /// Abstract method to perform backward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">The upstream gradient.</param>
        /// <returns>The gradients to send to the adjacent backward operations.</returns>
        public abstract BackwardResult[] Backward(DeepMatrix dOutput);
    }
}
