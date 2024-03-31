//------------------------------------------------------------------------------
// <copyright file="PairwiseSineSoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Sine Softmax operation.
    /// </summary>
    public class PairwiseSineSoftmaxOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PairwiseSineSoftmaxOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.Output }, (x, y) => new[] { this.input, this.Output });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.Output = restored[1];
        }

        /// <summary>
        /// Performs the forward operation for the softmax function.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <returns>The output of the softmax operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            this.Output = this.PairedSineSoftmax(input);
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = this.Output.Length;
            int numCols = this.Output[0].Length;
            int m = numCols / 2;

            Matrix dLdInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    double a = this.input[i, j];
                    double b = this.input[i, j + m];
                    double expSinA = Math.Exp(Math.Sin(a));
                    double expSinB = Math.Exp(Math.Sin(b));

                    double f1 = expSinA;
                    double fPrime1 = Math.Cos(a) * expSinA;
                    double g1 = expSinA + expSinB;
                    double gPrime1 = Math.Cos(a) * expSinA;
                    double dSinSoftmaxj1 = ((fPrime1 * g1) - (f1 * gPrime1)) / Math.Pow(g1, 2d);

                    double f2 = expSinB;
                    double fPrime2 = Math.Cos(b) * expSinB;
                    double g2 = expSinA + expSinB;
                    double gPrime2 = Math.Cos(b) * expSinB;
                    double dSinSoftmaxj2 = ((fPrime2 * g2) - (f2 * gPrime2)) / Math.Pow(g2, 2d);

                    dLdInput[i][j] = dLdOutput[i][j] * dSinSoftmaxj1;
                    dLdInput[i][j + m] = dLdOutput[i][j + m] * dSinSoftmaxj2;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        private Matrix PairedSineSoftmax(Matrix input)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;
            int m = numCols / 2;

            Matrix output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    double a = input[i, j];
                    double b = input[i, j + m];
                    double sumExp = Math.Exp(Math.Sin(a)) + Math.Exp(Math.Sin(b));
                    double numerator1 = Math.Exp(Math.Sin(a));
                    output[i, j] = numerator1 / sumExp;
                    double numerator2 = Math.Exp(Math.Sin(b));
                    output[i, j + m] = numerator2 / sumExp;
                }
            }

            return output;
        }
    }
}
