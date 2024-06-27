//------------------------------------------------------------------------------
// <copyright file="MatrixAddScalarOperation.cs" author="ameritusweb" date="1/3/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A matrix add scalar operation.
    /// </summary>
    public class MatrixAddScalarOperation : Operation
    {
        private Matrix input;
        private float scalar;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAddScalarOperation();
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
            this.scalar = (float)restored[0];
        }

        /// <summary>
        /// Performs the forward operation for the matrix add scalar function.
        /// </summary>
        /// <param name="input">The first input to the matrix add scalar operation.</param>
        /// <param name="scalar">The second input to the matrix add scalar operation.</param>
        /// <returns>The output of the matrix add scalar operation.</returns>
        public Matrix Forward(Matrix input, float scalar)
        {
            this.scalar = scalar;
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.Output = new Matrix(rows, cols);

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    this.Output[i][j] = input[i][j] + this.scalar;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            // The gradient w.r.t the input is the same as the gradient of the output
            // since addition of a scalar doesn't change the gradient.
            return new BackwardResultBuilder()
                .AddInputGradient(dLdOutput)
                .Build();
        }
    }
}
