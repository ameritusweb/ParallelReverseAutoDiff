//------------------------------------------------------------------------------
// <copyright file="SoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Softmax operation.
    /// </summary>
    public class SoftmaxOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SoftmaxOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrices.AddOrUpdate(id, this.Output, (_, _) => this.Output);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Output = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the softmax function.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <returns>The output of the softmax operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            this.Output = this.Softmax(input);
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
                for (int j = 0; j < numCols; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < numCols; k++)
                    {
                        if (j == k)
                        {
                            sum += dLdOutput[i][k] * this.Output[i][j] * (1 - this.Output[i][j]);
                        }
                        else
                        {
                            sum -= dLdOutput[i][k] * this.Output[i][j] * this.Output[i][k];
                        }
                    }

                    dLdInput[i][j] = sum;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        private Matrix Softmax(Matrix input)
        {
            int numRows = input.Length;
            int numCols = input[0].Length;

            Matrix output = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                double max = input[i].Max();
                double sum = 0;
                for (int j = 0; j < numCols; j++)
                {
                    double exp = Math.Exp(input[i][j] - max);
                    sum += exp;
                    output[i][j] = exp;
                }

                for (int j = 0; j < numCols; j++)
                {
                    output[i][j] /= sum;
                }
            }

            return output;
        }
    }
}
