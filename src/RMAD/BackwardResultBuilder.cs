//------------------------------------------------------------------------------
// <copyright file="BackwardResultBuilder.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using ParallelReverseAutoDiff.PRAD;

    /// <summary>
    /// The backward result builder.
    /// </summary>
    public class BackwardResultBuilder
    {
        private readonly List<object> backwardResults = new List<object>();
        private readonly List<Type> resultTypes = new List<Type>();
        private int inputGradientCount;
        private int deepInputGradientCount;

        /// <summary>
        /// Add input gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddInputGradient(Matrix matrix)
        {
            this.inputGradientCount++;
            this.resultTypes.Add(typeof(Matrix));
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add input gradient to the backward result.
        /// </summary>
        /// <param name="tensor">The tensor to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddInputGradient(Tensor tensor)
        {
            this.inputGradientCount++;
            this.resultTypes.Add(typeof(Matrix));
            this.backwardResults.Add(tensor.ToMatrix());
            return this;
        }

        /// <summary>
        /// Add input gradient array to the backward result.
        /// </summary>
        /// <param name="matrix">The deep matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddInputGradientArray(DeepMatrix matrix)
        {
            foreach (var mat in matrix)
            {
                this.inputGradientCount++;
                this.resultTypes.Add(typeof(Matrix));
                this.backwardResults.Add(mat);
            }

            return this;
        }

        /// <summary>
        /// Add input gradient array to the backward result.
        /// </summary>
        /// <param name="matrix">The 3-D tensor to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddInputGradientArray(Tensor matrix)
        {
            var deepMatrix = matrix.ToDeepMatrix();
            foreach (var mat in deepMatrix)
            {
                this.inputGradientCount++;
                this.resultTypes.Add(typeof(Matrix));
                this.backwardResults.Add(mat);
            }

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
            this.resultTypes.Add(typeof(DeepMatrix));
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add deep input gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The 3-D tensor to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddDeepInputGradient(Tensor matrix)
        {
            this.deepInputGradientCount++;
            this.resultTypes.Add(typeof(DeepMatrix));
            this.backwardResults.Add(matrix.ToDeepMatrix());
            return this;
        }

        /// <summary>
        /// Add filters gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The deep matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddFiltersGradient(DeepMatrix[] matrix)
        {
            this.resultTypes.Add(typeof(DeepMatrix[]));
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add weight gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddWeightGradient(Matrix matrix)
        {
            this.resultTypes.Add(typeof(Matrix));
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Add scaling gradient to the backward result.
        /// </summary>
        /// <param name="matrix">The matrix to add.</param>
        /// <returns>The backward result builder.</returns>
        public BackwardResultBuilder AddScalingGradient(Matrix matrix)
        {
            this.resultTypes.Add(typeof(Matrix));
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
            this.resultTypes.Add(typeof(Matrix));
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
            this.resultTypes.Add(typeof(Matrix));
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
            this.resultTypes.Add(typeof(Matrix));
            this.backwardResults.Add(matrix);
            return this;
        }

        /// <summary>
        /// Builds the backward result.
        /// </summary>
        /// <returns>The backward result.</returns>
        public BackwardResult Build()
        {
            return new BackwardResult { Results = this.backwardResults.ToArray(), HasMultipleInputs = this.inputGradientCount > 1 || this.deepInputGradientCount > 1 };
        }
    }
}
