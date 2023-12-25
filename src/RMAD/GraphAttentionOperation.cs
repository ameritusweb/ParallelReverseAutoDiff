//------------------------------------------------------------------------------
// <copyright file="GraphAttentionOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A graph attention operation for a graph attention network.
    /// </summary>
    public class GraphAttentionOperation : Operation
    {
        private Matrix nodeFeatures;  // NxM matrix
        private Matrix adjacency;     // NxN matrix
        private Matrix weights;       // 1x(2M) matrix

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new GraphAttentionOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.nodeFeatures, this.adjacency, this.weights }, (x, y) => new[] { this.nodeFeatures, this.adjacency, this.weights });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.nodeFeatures = restored[0];
            this.adjacency = restored[1];
            this.weights = restored[2];
        }

        /// <summary>
        /// The forward pass of the graph attention operation.
        /// </summary>
        /// <param name="nodeFeatures">The coefficients.</param>
        /// <param name="adjacency">The features.</param>
        /// <param name="weights">The weights.</param>
        /// <returns>The attention scores.</returns>
        public Matrix Forward(Matrix nodeFeatures, Matrix adjacency, Matrix weights)
        {
            this.nodeFeatures = nodeFeatures;
            this.adjacency = adjacency;
            this.weights = weights;

            int n = nodeFeatures.Rows;  // Number of nodes
            int m = nodeFeatures.Cols;  // Number of features per node

            this.Output = new Matrix(n, n);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    // Check if nodes are adjacent
                    if (adjacency[i, j] != 0)
                    {
                        // Concatenate and transpose node features
                        Matrix concatenatedFeatures = new Matrix(2 * m, 1);
                        for (int k = 0; k < m; k++)
                        {
                            concatenatedFeatures[k, 0] = nodeFeatures[i, k];
                            concatenatedFeatures[m + k, 0] = nodeFeatures[j, k];
                        }

                        // Calculate attention coefficient
                        double attentionCoefficient = 0;
                        for (int k = 0; k < 2 * m; k++)
                        {
                            attentionCoefficient += weights[0, k] * concatenatedFeatures[k, 0];
                        }

                        this.Output[i, j] = attentionCoefficient;
                    }
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int n = this.nodeFeatures.Rows;  // Number of nodes
            int m = this.nodeFeatures.Cols;  // Number of features per node

            Matrix dNodeFeatures = new Matrix(n, m);
            Matrix dWeights = new Matrix(1, 2 * m);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    // Check if nodes are adjacent
                    if (this.adjacency[i, j] != 0)
                    {
                        // Calculate gradient w.r.t. node features
                        for (int k = 0; k < m; k++)
                        {
                            dNodeFeatures[i, k] += dOutput[i, j] * this.weights[0, k];         // Gradient for features of node i
                            dNodeFeatures[j, k] += dOutput[i, j] * this.weights[0, m + k];    // Gradient for features of node j
                        }

                        // Calculate gradient w.r.t. weights
                        for (int k = 0; k < m; k++)
                        {
                            dWeights[0, k] += dOutput[i, j] * this.nodeFeatures[i, k];         // Gradient for the first half of weights
                            dWeights[0, m + k] += dOutput[i, j] * this.nodeFeatures[j, k];    // Gradient for the second half of weights
                        }
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dNodeFeatures)
                .AddWeightGradient(dWeights)
                .Build();
        }
    }
}
