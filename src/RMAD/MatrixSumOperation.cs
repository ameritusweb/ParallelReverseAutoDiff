//------------------------------------------------------------------------------
// <copyright file="MatrixSumOperation.cs" author="ameritusweb" date="6/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Matrix sum operation.
    /// </summary>
    public class MatrixSumOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixSumOperation();
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
        /// Performs the forward operation for the matrix sum function.
        /// </summary>
        /// <param name="input">The input to the matrix sum operation, an NxM matrix.</param>
        /// <returns>The output of the matrix sum operation, an Nx1 matrix.</returns>
        public Matrix Forward(Matrix input)
        {
            int numRows = input.Rows;
            this.input = input;
            this.Output = new Matrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                this.Output[i][0] = input[i].Sum();
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Rows;
            int numCols = this.input.Cols;
            Matrix dInput = new Matrix(numRows, numCols);

            // The gradient with respect to the input is the gradient of the output distributed equally across each element of the input row
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInput[i][j] = dOutput[i][0];
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}
