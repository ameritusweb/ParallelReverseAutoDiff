//------------------------------------------------------------------------------
// <copyright file="ElementwiseMultiplyAndSumOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Custom element-wise multiplication and summation operation.
    /// </summary>
    public class ElementwiseMultiplyAndSumOperation : Operation
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
            return new ElementwiseMultiplyAndSumOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            // Store the intermediate matrices
            this.IntermediateMatrices.AddOrUpdate(id, this.a, (_, _) => this.a);
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.b, (_, _) => this.b);
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
                Matrix elementwiseMultiplied = a.ElementwiseMultiply(slice);
                double sum = elementwiseMultiplied.Sum();
                this.Output[0][q] = sum;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int p = this.b.Depth;
            DeepMatrix dInputB = new DeepMatrix(p, this.a.Rows, this.a.Cols);

            for (int q = 0; q < p; q++)
            {
                Matrix dSlice = this.a * dOutput[0][q];  // Element-wise multiplication with a scalar
                dInputB[q] = dSlice;
            }

            Matrix dInputA = new Matrix(this.a.Rows, this.a.Cols);

            for (int q = 0; q < p; q++)
            {
                dInputA = dInputA + (this.b[q] * dOutput[0][q]);
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputA)
                .AddInputGradientArray(dInputB)
                .Build();
        }
    }
}