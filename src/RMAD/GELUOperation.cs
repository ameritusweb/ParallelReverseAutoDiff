//------------------------------------------------------------------------------
// <copyright file="GELUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Performs the forward and backward operations for the GELU activation function.
    /// </summary>
    public class GELUOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new GELUOperation();
        }

        /// <summary>
        /// Performs the forward operation for the GELU activation function.
        /// </summary>
        /// <param name="input">The input to the GELU operation.</param>
        /// <returns>The output of the GELU operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int rows = input.Rows;
            int cols = input.Cols;

            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i, j];
                    double gelu = this.GELU(x);
                    this.Output[i, j] = gelu;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Rows;
            int cols = dLdOutput.Cols;
            Matrix dLdInput = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i, j];
                    dLdInput[i, j] = dLdOutput[i, j] * this.DerivativeGELU(x);
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        private double GELU(double x)
        {
            double a = 0.5 * (1.0 + Math.Tanh(
                Math.Sqrt(2 / Math.PI) * (x + (0.044715 * Math.Pow(x, 3)))));

            return a * x;
        }

        private double DerivativeGELU(double x)
        {
            double a = (0.0356774 * Math.Pow(x, 3)) + (0.797885 * x);
            double b = Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + (0.044715 * Math.Pow(x, 3))));
            double c = Math.Pow(1 - Math.Pow(b, 2), 2);
            double d = Math.Sqrt(2 / Math.PI) * ((0.044715 * 3 * Math.Pow(x, 2)) + 1);

            return (0.5 * x * ((c * d) + b)) + (0.5 * (1 + b));
        }
    }
}
