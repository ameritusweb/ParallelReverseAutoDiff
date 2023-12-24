//------------------------------------------------------------------------------
// <copyright file="RMSNormOperation.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// RMSNorm operation for a Matrix.
    /// </summary>
    public class RMSNormOperation : Operation
    {
        private Matrix input;
        private Matrix g;  // Scaling factor, a learnable parameter

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new RMSNormOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.g }, (x, y) => new[] { this.input, this.g });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.g = restored[1];
        }

        /// <summary>
        /// The forward pass of the RMSNorm operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="g">The scaling factor.</param>
        /// <returns>The output matrix.</returns>
        public Matrix Forward(Matrix input, Matrix g)
        {
            this.input = input;
            this.g = g;
            int rows = input.Rows;
            int cols = input.Cols;

            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                double sumSq = 0;
                for (int j = 0; j < cols; j++)
                {
                    sumSq += Math.Pow(input[i, j], 2);
                }

                double rms = Math.Sqrt(sumSq / cols);

                for (int j = 0; j < cols; j++)
                {
                    this.Output[i, j] = (input[i, j] / rms) * this.g[i, j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix gradOutput)
        {
            int rows = this.input.Rows;
            int cols = this.input.Cols;

            Matrix dx = new Matrix(rows, cols);
            Matrix dg = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                double sumSq = 0;
                for (int j = 0; j < cols; j++)
                {
                    sumSq += Math.Pow(this.input[i, j], 2);
                }

                double rms = Math.Sqrt(sumSq / cols);

                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i, j];
                    double g = this.g[i, j];

                    dx[i, j] = (g / rms) - ((x * g * x) / (cols * Math.Pow(rms, 3)));
                    dg[i, j] = x / rms;

                    dx[i, j] *= gradOutput[i, j]; // Chain rule
                    dg[i, j] *= gradOutput[i, j]; // Chain rule
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dx)
                .AddScalingGradient(dg)
                .Build();
        }
    }
}