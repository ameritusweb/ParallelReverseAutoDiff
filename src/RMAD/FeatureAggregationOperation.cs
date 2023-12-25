//------------------------------------------------------------------------------
// <copyright file="FeatureAggregationOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A feature aggregation operation for a graph attention network.
    /// </summary>
    public class FeatureAggregationOperation : Operation
    {
        private Matrix coefficients; // NxN matrix
        private Matrix features; // NxP matrix

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static FeatureAggregationOperation Instantiate(NeuralNetwork net)
        {
            return new FeatureAggregationOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.coefficients, this.features }, (x, y) => new[] { this.coefficients, this.features });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.coefficients = restored[0];
            this.features = restored[1];
        }

        /// <summary>
        /// The forward pass of the embedding operation.
        /// </summary>
        /// <param name="coefficients">The coefficients.</param>
        /// <param name="features">The features.</param>
        /// <returns>The aggregated features.</returns>
        public Matrix Forward(Matrix coefficients, Matrix features)
        {
            if (coefficients.Rows != coefficients.Cols || coefficients.Rows != features.Rows)
            {
                throw new ArgumentException("Invalid dimensions for coefficient and feature matrices.");
            }

            int n = features.Rows; // Number of nodes
            int p = features.Cols; // Number of features per node

            Matrix aggregatedFeatures = new Matrix(n, p);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    double coefficient = coefficients[i, j];
                    for (int k = 0; k < p; k++)
                    {
                        // Aggregate features based on the coefficients
                        aggregatedFeatures[i][k] += coefficient * features[j][k];
                    }
                }
            });

            this.Output = aggregatedFeatures;
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int n = this.features.Rows; // Number of nodes
            int p = this.features.Cols; // Number of features per node

            // Initialize gradients
            Matrix dCoefficients = new Matrix(n, n);
            Matrix dFeatures = new Matrix(n, p);

            // Calculate the gradient with respect to coefficients and features
            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < p; k++)
                    {
                        // Gradient with respect to the coefficient
                        dCoefficients[i][j] += dOutput[i][k] * this.features[j][k];

                        // Gradient with respect to the features
                        dFeatures[j][k] += dOutput[i][k] * this.coefficients[i, j];
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dCoefficients)
                .AddInputGradient(dFeatures)
                .Build();
        }
    }
}
