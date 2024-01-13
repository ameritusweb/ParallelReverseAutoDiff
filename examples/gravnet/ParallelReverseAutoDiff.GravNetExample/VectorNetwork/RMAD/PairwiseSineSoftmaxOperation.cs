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
            int M = numCols / 2;

            Matrix dLdInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    double a = this.input[i, j];
                    double b = this.input[i, j + M];
                    double expSinA = Math.Exp(Math.Sin(a));
                    double expSinB = Math.Exp(Math.Sin(b));

                    double g1 = (2 * expSinA) + expSinB;
                    // double dSinSoftmaxj1 = ((expSinA * Math.Cos(a) * g1) - (expSinA * 2 * expSinA * Math.Cos(a))) / Math.Pow(g1, 2d);
                    // double dSinSoftmaxj1 = (expSinA * Math.Cos(a)) * (g1 - (2 * expSinA)) / Math.Pow(g1, 2d);
                    double dSinSoftmaxj1 = (expSinA * Math.Cos(a)) * expSinB / Math.Pow(g1, 2d);
                    double g2 = expSinA + (2 * expSinB);
                    // double dSinSoftmaxj2 = ((expSinB * Math.Cos(b) * g2) - (expSinB * 2 * expSinB * Math.Cos(b))) / Math.Pow(g2, 2d);
                    // double dSinSoftmaxj2 = (expSinB * Math.Cos(b)) * (g2 - (2 * expSinB)) / Math.Pow(g2, 2d);
                    double dSinSoftmaxj2 = (expSinB * Math.Cos(b)) * expSinA / Math.Pow(g2, 2d);
                    dLdInput[i][j] = dLdOutput[i][j] * dSinSoftmaxj1;
                    dLdInput[i][j + M] = dLdOutput[i][j + M] * dSinSoftmaxj2;
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
            int M = numCols / 2;

            Matrix output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    double a = input[i, j];
                    double b = input[i, j + M];
                    double sumExp = Math.Exp(Math.Sin(a)) + Math.Exp(Math.Sin(b));
                    double numerator1 = Math.Exp(Math.Sin(a));
                    output[i, j] = numerator1 / (sumExp + numerator1);
                    double numerator2 = Math.Exp(Math.Sin(b));
                    output[i, j + M] = numerator2 / (sumExp + numerator2);
                }
            }

            return output;
        }
    }
}
