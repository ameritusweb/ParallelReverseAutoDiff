﻿//------------------------------------------------------------------------------
// <copyright file="StretchedSigmoidOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A stretched sigmoid operation.
    /// </summary>
    public class StretchedSigmoidOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new StretchedSigmoidOperation();
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
        /// Performs the forward operation for the stretched sigmoid activation function.
        /// </summary>
        /// <param name="input">The input to the stretched sigmoid operation.</param>
        /// <returns>The output of the stretched sigmoid operation.</returns>
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
                    this.Output[i][j] = 1.0f / (1.0f + PradMath.Pow(PradMath.PI - 2, -input[i][j]));
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = dLdOutput.Length;
            int numCols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    var x = this.input[i][j];
                    var dx = PradMath.Pow(PradMath.PI - 2, -x) * PradMath.Log(PradMath.PI - 2) / PradMath.Pow(1 + PradMath.Pow(PradMath.PI - 2, -x), 2);
                    dLdInput[i][j] = dLdOutput[i][j] * dx;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
