//------------------------------------------------------------------------------
// <copyright file="IOptimizer.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// Optimizes weight updates.
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        /// Gets a value indicating whether the optimizer is initialized.
        /// </summary>
        bool IsInitialized { get; }

        /// <summary>
        /// Initializes the momentum.
        /// </summary>
        /// <param name="parameter">The parameter.</param>
        void Initialize(Tensor parameter);    // Initializes any state (momentum, etc.) related to the parameter

        /// <summary>
        /// Updates the weights.
        /// </summary>
        /// <param name="weights">The weights to update.</param>
        /// <param name="gradient">The gradient.</param>
        void UpdateWeights(Tensor weights, Tensor gradient);  // Updates weights using the provided gradient
    }
}
