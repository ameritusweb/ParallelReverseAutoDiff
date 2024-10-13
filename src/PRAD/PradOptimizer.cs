//------------------------------------------------------------------------------
// <copyright file="PradOptimizer.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// Optimizes the update of weights.
    /// </summary>
    public static class PradOptimizer
    {
        /// <summary>
        /// Creates an Adam optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="beta1">The first beta.</param>
        /// <param name="beta2">The second beta.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <returns>The optimizer.</returns>
        public static IOptimizer CreateAdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
        }

        /// <summary>
        /// Creates an Adam optimizer with momentum.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="beta1">The first beta.</param>
        /// <param name="beta2">The second beta.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <returns>The optimizer.</returns>
        public static IOptimizer CreateAdamOptimizerWithMomentum(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            return new MomentumAdamOptimizer(learningRate, beta1, beta2, epsilon);
        }

        /// <summary>
        /// Creates an RMSProp Optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <returns>The optimizer.</returns>
        public static IOptimizer CreateRMSPropOptimizer(double learningRate = 0.001, double beta = 0.9, double epsilon = 1e-8)
        {
            return new RMSPropOptimizer(learningRate, beta, epsilon);
        }
    }
}
