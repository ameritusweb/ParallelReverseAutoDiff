//------------------------------------------------------------------------------
// <copyright file="TakeRightOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A take right operation.
    /// </summary>
    public class TakeRightOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static TakeRightOperation Instantiate(NeuralNetwork net)
        {
            return new TakeRightOperation();
        }

        /// <summary>
        /// Applies a take right operation to the input matrix.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix Forward(Matrix input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (input.Cols % 2 != 0)
            {
                throw new ArgumentException("The number of columns in the input matrix must be even.");
            }

            this.input = input;
            int numRows = input.Rows;
            int halfCols = input.Cols / 2;

            this.Output = new Matrix(numRows, halfCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < halfCols; j++)
                {
                    this.Output[i, j] = input[i, halfCols + j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = this.input.Rows;
            int halfCols = this.input.Cols / 2;

            Matrix dLdInput = new Matrix(numRows, this.input.Cols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < halfCols; j++)
                {
                    dLdInput[i, halfCols + j] = dLdOutput[i, j];
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
