//------------------------------------------------------------------------------
// <copyright file="AdamOptimizer.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Optimizes weight updates using the Adam algorithm.
    /// </summary>
    public class AdamOptimizer : IOptimizer
    {
        private Tensor m;  // First moment vector (mean of the gradients)
        private Tensor v;  // Second moment vector (variance of the gradients)
        private int t;     // Timestep

        /// <summary>
        /// Initializes a new instance of the <see cref="AdamOptimizer"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="beta1">The first beta parameter for the exponential decay rate of the first moment estimates.</param>
        /// <param name="beta2">The second beta parameter for the exponential decay rate of the second moment estimates.</param>
        /// <param name="epsilon">A small constant to prevent division by zero.</param>
        public AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
            this.t = 0;
        }

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Gets or sets the first beta (controls first moment decay).
        /// </summary>
        public double Beta1 { get; set; }

        /// <summary>
        /// Gets or sets the second beta (controls second moment decay).
        /// </summary>
        public double Beta2 { get; set; }

        /// <summary>
        /// Gets or sets epsilon (small value to prevent division by zero).
        /// </summary>
        public double Epsilon { get; set; }

        /// <summary>
        /// Initialize the optimizer's moment vectors.
        /// </summary>
        /// <param name="parameter">The tensor parameter representing the weights.</param>
        public void Initialize(Tensor parameter)
        {
            // Initialize moment vectors to zeros, matching the shape of the parameter (weights)
            this.m = new Tensor(parameter.Shape, 0.0);
            this.v = new Tensor(parameter.Shape, 0.0);
        }

        /// <summary>
        /// Update the weights using the Adam optimization algorithm.
        /// </summary>
        /// <param name="weights">The weights to be updated.</param>
        /// <param name="gradient">The gradient of the loss with respect to the weights.</param>
        public void UpdateWeights(Tensor weights, Tensor gradient)
        {
            this.t++;  // Increment timestep

            // Compute the biased first moment estimate (m = beta1 * m + (1 - beta1) * gradient)
            this.m = this.m.ElementwiseMultiply(new Tensor(this.m.Shape, this.Beta1))
                          .ElementwiseAdd(gradient.ElementwiseMultiply(new Tensor(gradient.Shape, PradTools.One - this.Beta1)));

            // Compute the biased second moment estimate (v = beta2 * v + (1 - beta2) * gradient^2)
            this.v = this.v.ElementwiseMultiply(new Tensor(this.v.Shape, this.Beta2))
                          .ElementwiseAdd(gradient.ElementwiseSquare().ElementwiseMultiply(new Tensor(gradient.Shape, PradTools.One - this.Beta2)));

            // Bias correction for the first moment (mHat = m / (1 - beta1^t))
            var mHat = this.m.ElementwiseDivide(new Tensor(this.m.Shape, PradTools.One - Math.Pow(this.Beta1, this.t)));

            // Bias correction for the second moment (vHat = v / (1 - beta2^t))
            var vHat = this.v.ElementwiseDivide(new Tensor(this.v.Shape, PradTools.One - Math.Pow(this.Beta2, this.t)));

            // Update the weights (weights -= learningRate * mHat / (sqrt(vHat) + epsilon))
            var weightUpdate = mHat.ElementwiseMultiply(new Tensor(mHat.Shape, this.LearningRate))
                                   .ElementwiseDivide(vHat.ElementwiseSquareRoot().ElementwiseAdd(new Tensor(vHat.Shape, this.Epsilon)));

            // Replace the data in the original weights tensor with the updated values
            weights.ReplaceData(weights.ElementwiseSub(weightUpdate).Data);
        }
    }
}
