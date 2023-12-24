//------------------------------------------------------------------------------
// <copyright file="MatrixMultiplyAndSumOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Custom matrix multiplication and summation operation.
    /// </summary>
    public class MatrixMultiplyAndSumOperation : Operation
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
            return new MatrixMultiplyAndSumOperation();
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
        /// <param name="a">NxN Matrix.</param>
        /// <param name="b">PxNxN DeepMatrix.</param>
        /// <returns>1xP Matrix.</returns>
        public Matrix Forward(Matrix a, DeepMatrix b)
        {
            this.a = a;
            this.b = b;

            int p = b.Depth;
            this.Output = new Matrix(1, p);

            for (int q = 0; q < p; q++)
            {
                Matrix slice = b[q];
                Matrix matrixMultiplied = a * slice;
                double sum = matrixMultiplied.Sum();
                this.Output[0][q] = sum;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int p = this.b.Depth;
            DeepMatrix dInputB = new DeepMatrix(p, this.a.Rows, this.a.Cols);

            Matrix dInputA = new Matrix(this.a.Rows, this.a.Cols);

            for (int q = 0; q < p; q++)
            {
                Matrix dSlice = this.a.Transpose() * dOutput[0][q]; // Multiply by scalar and then by matrix
                dInputB[q] = dSlice;

                dInputA = dInputA + (this.b[q].Transpose() * dOutput[0][q]); // Multiply by scalar and then by matrix
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputA)
                .AddInputGradientArray(dInputB)
                .Build();
        }
    }
}