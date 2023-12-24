//------------------------------------------------------------------------------
// <copyright file="MatrixMultiplyScalarOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A matrix multiply scalar operation.
    /// </summary>
    public class MatrixMultiplyScalarOperation : Operation
    {
        private Matrix input;
        private double scalar;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyScalarOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateObjectArrays.AddOrUpdate(id, new[] { (object)this.scalar }, (x, y) => new[] { (object)this.scalar });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.scalar = (double)restored[0];
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply scalar function.
        /// </summary>
        /// <param name="input">The first input to the matrix multiply scalar operation.</param>
        /// <param name="scalar">The second input to the matrix multiply scalar operation.</param>
        /// <returns>The output of the matrix multiply scalar operation.</returns>
        public Matrix Forward(Matrix input, double scalar)
        {
            this.scalar = scalar;
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.Output = new Matrix(rows, cols);

            // Parallelize the outer loop
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    this.Output[i][j] = input[i][j] * this.scalar;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(rows, cols);

            // Parallelize the outer loop
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    dLdInput[i][j] = dLdOutput[i][j] * this.scalar;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
