//------------------------------------------------------------------------------
// <copyright file="PradOperationBase.cs" author="ameritusweb" date="6/20/2024">
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
    public abstract class PradOperationBase<TOperation, TSubType, TReturnType> : OperationBase, IOperation
        where TOperation : PradOperationBase<TOperation, TSubType, TReturnType>
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
        public abstract BackwardResult Backward(Matrix dOutput);
    }

    /// <summary>
    /// A base class for PRAD operations.
    /// </summary>
    /// <typeparam name="TOperation">The operation type.</typeparam>
    /// <typeparam name="TSubType1">The first subtype.</typeparam>
    /// <typeparam name="TSubType2">The second subtype.</typeparam>
    /// <typeparam name="TReturnType">The return type.</typeparam>"
    [System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1402:File may only contain a single type", Justification = "Same name")]
    public abstract class PradOperationBase<TOperation, TSubType1, TSubType2, TReturnType> : OperationBase, IOperation
        where TOperation : PradOperationBase<TOperation, TSubType1, TSubType2, TReturnType>
    {
        /// <summary>
        /// Abstract method to perform forward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="input1">The first input to the forward pass.</param>
        /// <param name="input2">The second input to the forward pass.</param>
        /// <returns>The result of the forward pass.</returns>
        public abstract TReturnType Forward(TSubType1 input1, TSubType2 input2);

        /// <summary>
        /// Abstract method to perform backward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">The upstream gradient.</param>
        /// <returns>The gradients to send to the adjacent backward operations.</returns>
        public abstract BackwardResult Backward(Matrix dOutput);
    }
}
