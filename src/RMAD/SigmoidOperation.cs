//------------------------------------------------------------------------------
// <copyright file="SigmoidOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Sigmoid operation.
    /// </summary>
    public class SigmoidOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SigmoidOperation();
        }

        /// <summary>
        /// Performs the forward operation for the sigmoid activation function.
        /// </summary>
        /// <param name="input">The input to the sigmoid operation.</param>
        /// <returns>The output of the sigmoid operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            this.Output = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = 1.0 / (1.0 + Math.Exp(-input[i][j]));
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = this.input.Length;
            int numCols = this.input[0].Length;

            Matrix dInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    double sigmoidDerivative = this.Output[i][j] * (1 - this.Output[i][j]);
                    dInput[i][j] = dOutput[i][j] * sigmoidDerivative;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}
