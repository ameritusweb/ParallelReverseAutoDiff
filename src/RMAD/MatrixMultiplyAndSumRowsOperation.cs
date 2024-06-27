//------------------------------------------------------------------------------
// <copyright file="MatrixMultiplyAndSumRowsOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Custom matrix multiplication and summation operation.
    /// </summary>
    public class MatrixMultiplyAndSumRowsOperation : Operation
    {
        private Matrix a;
        private DeepMatrix b;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyAndSumRowsOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            // Store the intermediate matrices
            this.IntermediateMatrices.AddOrUpdate(id, this.a, (x, y) => this.a);
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.b, (x, y) => this.b);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            // Restore the intermediate matrices
            this.a = this.IntermediateMatrices[id];
            this.b = this.IntermediateDeepMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="a">MxN Matrix.</param>
        /// <param name="b">PxNxN DeepMatrix.</param>
        /// <returns>MxP Matrix.</returns>
        public Matrix Forward(Matrix a, DeepMatrix b)
        {
            this.a = a;
            this.b = b;

            int p = b.Depth;
            this.Output = new Matrix(a.Rows, p);

            for (int q = 0; q < p; q++)
            {
                Matrix slice = b[q];
                Matrix matrixMultiplied = a * slice;
                for (int i = 0; i < a.Rows; i++)
                {
                    this.Output[i][q] = matrixMultiplied[i].Sum();
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dA = new Matrix(this.a.Rows, this.a.Cols);
            DeepMatrix dB = new DeepMatrix(this.b.Dimension);

            for (int q = 0; q < this.b.Depth; q++)
            {
                Matrix slice = this.b[q];
                dB[q] = new Matrix(this.a.Cols, this.a.Cols); // Initialize to zero NxN Matrix
                dB[q].Initialize(InitializationType.Zeroes);
                for (int i = 0; i < this.a.Rows; i++)
                {
                    var multiplier = dOutput[i][q]; // This is a scalar
                    Matrix scaledSlice = slice * multiplier;  // This scales each element in slice

                    for (int j = 0; j < dA[i].Length; j++)
                    {
                        dA[i][j] += scaledSlice[0][j]; // scaledSlice is a row vector in matrix form
                    }

                    Matrix contributionToSlice = new Matrix(this.a[i]).Transpose() * new Matrix(this.a[i]) * multiplier; // a[i] gives the i-th row as a vector
                    dB[q] += contributionToSlice; // += overloads to add matrices element-wise
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dA)
                .AddInputGradientArray(dB)
                .Build();
        }
    }
}