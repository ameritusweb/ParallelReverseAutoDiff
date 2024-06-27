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
            int n = nodeFeatures.Rows;
            int m = nodeFeatures.Cols;
            this.Output = new Matrix(n, n);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    Matrix concatenatedFeatures = new Matrix(2 * m, 1);
                    for (int k = 0; k < m; k++)
                    {
                        concatenatedFeatures[k, 0] = nodeFeatures[i, k];
                        concatenatedFeatures[m + k, 0] = nodeFeatures[j, k];
                    }

                    float attentionCoefficient = 0;
                    for (int k = 0; k < 2 * m; k++)
                    {
                        attentionCoefficient += weights[0, k] * concatenatedFeatures[k, 0];
                    }

                    this.Output[i, j] = attentionCoefficient * adjacency[i, j];
                }
            });
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int n = this.nodeFeatures.Rows;
            int m = this.nodeFeatures.Cols;
            Matrix dNodeFeatures = new Matrix(n, m);
            Matrix dWeights = new Matrix(1, 2 * m);
            Matrix dAdjacency = new Matrix(n, n);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < m; k++)
                    {
                        var intermediate1 = dOutput[i, j] * this.weights[0, k];
                        var intermediate2 = dOutput[i, j] * this.weights[0, m + k];
                        var intermediate3 = dOutput[i, j] * this.nodeFeatures[i, k];
                        var intermediate4 = dOutput[i, j] * this.nodeFeatures[j, k];

                        if (float.IsNaN(intermediate1) || float.IsInfinity(intermediate1))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in intermediate1: {dOutput[i, j]} {this.weights[0, k]}");
                        }

                        if (float.IsNaN(intermediate2) || float.IsInfinity(intermediate2))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in intermediate2: {dOutput[i, j]} {this.weights[0, m + k]}");
                        }

                        if (float.IsNaN(intermediate3) || float.IsInfinity(intermediate3))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in intermediate3: {dOutput[i, j]} {this.nodeFeatures[i, k]}");
                        }

                        if (float.IsNaN(intermediate4) || float.IsInfinity(intermediate4))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in intermediate4: {dOutput[i, j]} {this.nodeFeatures[j, k]}");
                        }

                        dNodeFeatures[i, k] += intermediate1;
                        dNodeFeatures[j, k] += intermediate2;
                        dWeights[0, k] += intermediate3;
                        dWeights[0, m + k] += intermediate4;
                    }

                    float attentionGradient = 0;
                    for (int k = 0; k < 2 * m; k++)
                    {
                        attentionGradient += this.weights[0, k] * (k < m ? this.nodeFeatures[i, k % m] : this.nodeFeatures[j, k % m]);
                    }

                    dAdjacency[i, j] = dOutput[i, j] * attentionGradient;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dNodeFeatures)
                .AddInputGradient(dAdjacency)
                .AddWeightGradient(dWeights)
                .Build();
        }
    }
}
