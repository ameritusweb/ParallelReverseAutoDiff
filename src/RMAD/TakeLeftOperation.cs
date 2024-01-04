//------------------------------------------------------------------------------
// <copyright file="TakeLeftOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A take left operation.
    /// </summary>
    public class TakeLeftOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static TakeLeftOperation Instantiate(NeuralNetwork net)
        {
            return new TakeLeftOperation();
        }

        /// <summary>
        /// Applies a take left operation to the input matrix.
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
            int halfCols = input.Cols / 2;

            this.Output = new Matrix(1, halfCols);

            for (int j = 0; j < halfCols; j++)
            {
                this.Output[0, j] = input[0, j];
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
                    dLdInput[i, j] = dLdOutput[i, j];
                }

                // The rest of the matrix remains initialized to zero
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
