//------------------------------------------------------------------------------
// <copyright file="RMSPropOptimizer.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Optimizes weight updates.
    /// </summary>
    public class RMSPropOptimizer : IOptimizer
    {
        private Tensor s;  // Running average of squared gradients

        /// <summary>
        /// Initializes a new instance of the <see cref="RMSPropOptimizer"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="epsilon">The epsilon.</param>
        public RMSPropOptimizer(double learningRate = 0.001, double beta = 0.9, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Beta = beta;
            this.Epsilon = epsilon;
        }

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Gets or sets the beta.
        /// </summary>
        public double Beta { get; set; }

        /// <summary>
        /// Gets or sets the epsilon.
        /// </summary>
        public double Epsilon { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether is initialized.
        /// </summary>
        public bool IsInitialized { get; set; }

        /// <summary>
        /// Initializes the optimizer.
        /// </summary>
        /// <param name="parameter">The tensor parameter.</param>
        public void Initialize(Tensor parameter)
        {
            this.s = new Tensor(parameter.Shape, PradTools.Zero);  // Initialize running average to zeros
            this.IsInitialized = true;
        }

        /// <summary>
        /// Updates the weights.
        /// </summary>
        /// <param name="weights">The weights.</param>
        /// <param name="gradient">The gradient.</param>
        public void UpdateWeights(Tensor weights, Tensor gradient)
        {
            // Update running average of squared gradients
            this.s = this.s.ElementwiseMultiply(new Tensor(this.s.Shape, PradTools.Cast(this.Beta))).ElementwiseAdd(gradient.ElementwiseSquare().ElementwiseMultiply(new Tensor(gradient.Shape, PradTools.One - PradTools.Cast(this.Beta))));

            // Update weights using RMSProp's update rule
            weights.ReplaceData(weights.ElementwiseSub(gradient.ElementwiseMultiply(new Tensor(gradient.Shape, PradTools.Cast(this.LearningRate)))
                .ElementwiseDivide(this.s.ElementwiseSquareRoot().ElementwiseAdd(new Tensor(this.s.Shape, PradTools.Cast(this.Epsilon))))).Data);
        }
    }
}
