//------------------------------------------------------------------------------
// <copyright file="SineSoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Sine Softmax operation.
    /// </summary>
    public class SineSoftmaxOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SineSoftmaxOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.Output }, (_, _) => new[] { this.input, this.Output });
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
            this.Output = this.SineSoftmax(input);
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = this.Output.Length;
            int numCols = this.Output[0].Length;

            Matrix dLdInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                double mM = 0.0;
                for (int k = 0; k < numCols; k++)
                {
                    mM += Math.Exp(Math.Sin(this.input[i, k]));
                }

                for (int j = 0; j < numCols; j++)
                {
                    double dSinSoftmaxj = (mM * Math.Exp(Math.Sin(this.input[i, j])) * Math.Cos(this.input[i, j])) / Math.Pow(mM + Math.Exp(Math.Sin(this.input[i, j])), 2);
                    dLdInput[i][j] = dLdOutput[i][j] * dSinSoftmaxj;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        private Matrix SineSoftmax(Matrix input)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;

            Matrix output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                double mM = 0.0;

                for (int j = 0; j < numCols; j++)
                {
                    mM += Math.Exp(Math.Sin(input[i, j]));
                }

                for (int j = 0; j < numCols; j++)
                {
                    double numerator = Math.Exp(Math.Sin(input[i, j]));
                    output[i, j] = numerator / (mM + numerator);
                }
            }

            return output;
        }
    }
}
