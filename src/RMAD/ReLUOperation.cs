﻿//------------------------------------------------------------------------------
// <copyright file="ReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Performs the forward and backward operations for the ReLU activation function.
    /// </summary>
    public class ReLUOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ReLUOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrices.AddOrUpdate(id, this.input, (x, y) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the ReLU activation function.
        /// </summary>
        /// <param name="input">The input to the ReLU operation.</param>
        /// <returns>The output of the ReLU operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    var x = input[i][j];
                    this.Output[i][j] = x > 0 ? x : 0;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    var x = this.input[i][j];
                    var gradient = x > 0 ? 1.0f : 0.0f;
                    dLdInput[i][j] = dLdOutput[i][j] * gradient;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
