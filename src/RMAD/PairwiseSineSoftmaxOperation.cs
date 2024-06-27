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
                    var a = this.input[i, j];
                    var b = this.input[i, j + m];
                    var expSinA = PradMath.Exp(PradMath.Sin(a));
                    var expSinB = PradMath.Exp(PradMath.Sin(b));

                    var f1 = expSinA;
                    var fPrime1 = PradMath.Cos(a) * expSinA;
                    var g1 = expSinA + expSinB;
                    var gPrime1 = PradMath.Cos(a) * expSinA;
                    var dSinSoftmaxj1 = ((fPrime1 * g1) - (f1 * gPrime1)) / PradMath.Pow(g1, 2f);

                    var f2 = expSinB;
                    var fPrime2 = PradMath.Cos(b) * expSinB;
                    var g2 = expSinA + expSinB;
                    var gPrime2 = PradMath.Cos(b) * expSinB;
                    var dSinSoftmaxj2 = ((fPrime2 * g2) - (f2 * gPrime2)) / PradMath.Pow(g2, 2f);

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
                    var a = input[i, j];
                    var b = input[i, j + m];
                    var sumExp = PradMath.Exp(PradMath.Sin(a)) + PradMath.Exp(PradMath.Sin(b));
                    var numerator1 = PradMath.Exp(PradMath.Sin(a));
                    output[i, j] = numerator1 / sumExp;
                    var numerator2 = PradMath.Exp(PradMath.Sin(b));
                    output[i, j + m] = numerator2 / sumExp;
                }
            }

            return output;
        }
    }
}
