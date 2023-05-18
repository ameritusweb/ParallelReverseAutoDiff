//------------------------------------------------------------------------------
// <copyright file="TanhOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// The tanh operation.
    /// </summary>
    public class TanhOperation : Operation
    {
        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new TanhOperation();
        }

        /// <summary>
        /// Performs the forward operation for the Tanh activation function.
        /// </summary>
        /// <param name="input">The input to the Tanh operation.</param>
        /// <returns>The output of the Tanh operation.</returns>
        public Matrix Forward(Matrix input)
        {
            int numRows = input.Length;
            int numCols = input[0].Length;

            this.Output = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = Math.Tanh(input[i][j]);
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;

            Matrix dInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    double derivative = 1 - Math.Pow(this.Output[i][j], 2);
                    dInput[i][j] = dOutput[i][j] * derivative;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}
