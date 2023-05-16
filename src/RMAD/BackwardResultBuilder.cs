//------------------------------------------------------------------------------
// <copyright file="BackwardResultBuilder.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The backward result builder.
    /// </summary>
    public class BackwardResultBuilder
    {
        private List<object> backwardResults = new List<object>();
        private int inputGradientCount = 0;
        private int deepInputGradientCount = 0;

        /// <summary>
        /// Add input gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddInputGradient(Matrix matrix)
        {
            this.inputGradientCount++;
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add deep input gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The deep matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddDeepInputGradient(DeepMatrix matrix)
        {
            this.deepInputGradientCount++;
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add filters gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The deep matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddFiltersGradient(DeepMatrix matrix)
        {
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add beta gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddBetaGradient(Matrix matrix)
        {
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add gamma gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddGammaGradient(Matrix matrix)
        {
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add bias gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddBiasGradient(Matrix matrix)
        {
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Builds the backward result.
        /// </summary>
        /// <returns>The backward result.</returns>
        public BackwardResult Build()
        {
            return new BackwardResult() { Results = this.backwardResults.ToArray(), HasMultipleInputs = this.inputGradientCount > 1 || this.deepInputGradientCount > 1 };
        }
    }
}
