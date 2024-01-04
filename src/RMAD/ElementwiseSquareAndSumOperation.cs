//------------------------------------------------------------------------------
// <copyright file="ElementwiseSquareAndSumOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Custom element-wise square and summation operation.
    /// </summary>
    public class ElementwiseSquareAndSumOperation : Operation
    {
        private Matrix a;
        private Matrix b;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseSquareAndSumOperation();
        }

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="a">1xN Matrix A.</param>
        /// <param name="b">1xN Matrix B.</param>
        /// <returns>Resultant 1x2 Matrix.</returns>
        public Matrix Forward(Matrix a, Matrix b)
        {
            if (a.Cols != b.Cols)
            {
                throw new ArgumentException("Both input matrices must have the same number of columns.");
            }

            this.a = a;
            this.b = b;

            double sum1 = 0;
            double sum2 = 0;

            for (int i = 0; i < a.Cols; i++)
            {
                sum1 += Math.Pow(a[0][i], 2);
                sum2 += Math.Pow(b[0][i], 2);
            }

            this.Output = new Matrix(new double[][] { new double[] { sum1, sum2 } });
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            if (dOutput.Cols != 2)
            {
                throw new ArgumentException("The gradient should be a 1x2 matrix.");
            }

            double dOutput1 = dOutput[0][0]; // Gradient for the first sum
            double dOutput2 = dOutput[0][1]; // Gradient for the second sum

            Matrix dInput1 = new Matrix(1, this.a.Cols);
            Matrix dInput2 = new Matrix(1, this.b.Cols);

            // Compute the gradient for each element of the input matrices
            for (int i = 0; i < this.a.Cols; i++)
            {
                dInput1[0][i] = 2 * this.a[0][i] * dOutput1; // Gradient for input1
                dInput2[0][i] = 2 * this.b[0][i] * dOutput2; // Gradient for input2
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .Build();
        }
    }
}