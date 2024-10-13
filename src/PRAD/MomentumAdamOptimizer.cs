//------------------------------------------------------------------------------
// <copyright file="MomentumAdamOptimizer.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Momentum Adam optimizer with additional tracking of positive and negative changes.
    /// </summary>
    public class MomentumAdamOptimizer : IOptimizer
    {
        private Tensor m;  // First moment vector
        private Tensor v;  // Second moment vector
        private int t;     // Timestep

        // Internal dictionaries for tracking changes
        private Dictionary<double, (int I, int J)> positiveChanges = new Dictionary<double, (int I, int J)>();
        private Dictionary<double, (int I, int J)> negativeChanges = new Dictionary<double, (int I, int J)>();
        private Dictionary<string, int> counts = new Dictionary<string, int>();
        private Dictionary<string, (int count, double value)> countValues = new Dictionary<string, (int, double)>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MomentumAdamOptimizer"/> class.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="beta1">The first beta parameter.</param>
        /// <param name="beta2">The second beta parameter.</param>
        /// <param name="epsilon">The epsilon parameter.</param>
        public MomentumAdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
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
        /// Gets or sets the first beta.
        /// </summary>
        public double Beta1 { get; set; }

        /// <summary>
        /// Gets or sets the second beta.
        /// </summary>
        public double Beta2 { get; set; }

        /// <summary>
        /// Gets or sets epsilon.
        /// </summary>
        public double Epsilon { get; set; }

        /// <summary>
        /// Initializes the optimizer for a given weight tensor.
        /// </summary>
        /// <param name="parameter">The weight tensor to initialize for.</param>
        public void Initialize(Tensor parameter)
        {
            this.m = new Tensor(parameter.Shape, 0.0);  // First moment vector
            this.v = new Tensor(parameter.Shape, 0.0);  // Second moment vector
        }

        /// <summary>
        /// Updates the weights using the momentum-based Adam optimization algorithm.
        /// </summary>
        /// <param name="weights">The weights tensor to update.</param>
        /// <param name="gradient">The gradient tensor for the current step.</param>
        public void UpdateWeights(Tensor weights, Tensor gradient)
        {
            this.positiveChanges.Clear();
            this.negativeChanges.Clear();
            this.t++;  // Increment the timestep

            // Update biased first moment estimate (m = beta1 * m + (1 - beta1) * gradient)
            var firstMoment = this.m.ElementwiseMultiply(new Tensor(this.m.Shape, this.Beta1))
                                    .ElementwiseAdd(gradient.ElementwiseMultiply(new Tensor(gradient.Shape, 1 - this.Beta1)));

            // Update biased second moment estimate (v = beta2 * v + (1 - beta2) * gradient^2)
            var secondMoment = this.v.ElementwiseMultiply(new Tensor(this.v.Shape, this.Beta2))
                                    .ElementwiseAdd(gradient.ElementwiseSquare()
                                    .ElementwiseMultiply(new Tensor(gradient.Shape, 1 - this.Beta2)));

            // Compute bias-corrected first moment estimate
            var mHat = firstMoment.ElementwiseMultiply(new Tensor(firstMoment.Shape, 1 / (1 - Math.Pow(this.Beta1, this.t))));

            // Compute bias-corrected second moment estimate
            var vHat = secondMoment.ElementwiseMultiply(new Tensor(secondMoment.Shape, 1 / (1 - Math.Pow(this.Beta2, this.t))));

            // Update weights with tracking logic
            for (int i = 0; i < weights.Shape[0]; i++)
            {
                for (int j = 0; j < weights.Shape[1]; j++)
                {
                    double weightReductionValue = this.LearningRate * mHat[i, j] / (Math.Sqrt(vHat[i, j]) + this.Epsilon);

                    string key = $"{i} {j}";
                    this.TrackWeightChange(weights, i, j, weightReductionValue, key);

                    // Apply weight reduction
                    weights[i, j] -= weightReductionValue;
                }
            }

            // Update m and v tensors in-place after weight update
            this.m.ReplaceData(firstMoment.Data);
            this.v.ReplaceData(secondMoment.Data);
        }

        /// <summary>
        /// Tracks weight changes and updates counts.
        /// </summary>
        private void TrackWeightChange(Tensor weights, int i, int j, double weightReductionValue, string key)
        {
            if (weights[i, j] > 0.0d)
            {
                if (!this.positiveChanges.ContainsKey(weightReductionValue))
                {
                    this.positiveChanges.Add(weightReductionValue, (i, j));
                }
            }
            else if (weights[i, j] < 0.0d)
            {
                if (!this.negativeChanges.ContainsKey(weightReductionValue))
                {
                    this.negativeChanges.Add(weightReductionValue, (i, j));
                }
            }

            // Update count tracking for momentum adjustments
            if (weightReductionValue > 0.0d)
            {
                if (this.counts.ContainsKey(key))
                {
                    this.counts[key] = Math.Max(0, this.counts[key] + 1);
                }
                else
                {
                    this.counts[key] = 1;
                }
            }
            else if (weightReductionValue < 0.0d)
            {
                if (this.counts.ContainsKey(key))
                {
                    this.counts[key] = Math.Min(0, this.counts[key] - 1);
                }
                else
                {
                    this.counts[key] = -1;
                }
            }

            // Apply decay logic if the count hits a threshold
            this.ApplyCountDecay(weights, i, j, weightReductionValue, key);
        }

        /// <summary>
        /// Applies count decay to the weight reduction value if certain conditions are met.
        /// </summary>
        private void ApplyCountDecay(Tensor weights, int i, int j, double weightReductionValue, string key)
        {
            if (this.counts.ContainsKey(key) && (this.counts[key] >= 2 || this.counts[key] <= -2))
            {
                if (!this.countValues.ContainsKey(key))
                {
                    this.countValues.Add(key, (this.counts[key], weights[i, j]));
                }
                else
                {
                    this.countValues[key] = (this.counts[key], weights[i, j]);
                }

                int countValue = this.counts[key];
                int absCountValue = Math.Min(75, Math.Abs(countValue));
                double decayFactor = Math.Pow(0.999d, absCountValue);
                double newValue = weightReductionValue * absCountValue * decayFactor;

                if (Math.Abs(newValue) < this.LearningRate)
                {
                    weights[i, j] -= newValue;
                }
            }
        }
    }
}
