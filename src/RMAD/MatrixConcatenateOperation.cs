//------------------------------------------------------------------------------
// <copyright file="MatrixConcatenateOperation.cs" author="ameritusweb" date="6/13/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Matrix concatenate operation.
    /// </summary>
    public class MatrixConcatenateOperation : Operation
    {
        private DeepMatrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixConcatenateOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.input, (x, y) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateDeepMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the matrix concatenate function.
        /// </summary>
        /// <param name="input">An array of matrices to concatenate. All matrices must have the same number of rows.</param>
        /// <returns>The output of the matrix concatenate operation.</returns>
        public Matrix Forward(DeepMatrix input)
        {
            this.input = input;
            int totalCols = input.Sum(m => m.Cols);
            int numRows = input[0].Rows;

            this.Output = new Matrix(numRows, totalCols);
            int currentCol = 0;

            foreach (Matrix matrix in input)
            {
                if (matrix.Rows != numRows)
                {
                    throw new ArgumentException("All input matrices must have the same number of rows.");
                }

                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < matrix.Cols; j++)
                    {
                        this.Output[i][currentCol + j] = matrix[i][j];
                    }
                }

                currentCol += matrix.Cols;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;
            DeepMatrix dInput = new DeepMatrix(this.input.Dimension);

            int currentCol = 0;
            for (int matrixIndex = 0; matrixIndex < this.input.Depth; matrixIndex++)
            {
                Matrix inputMatrix = this.input[matrixIndex];
                dInput[matrixIndex] = new Matrix(inputMatrix.Rows, inputMatrix.Cols);

                for (int i = 0; i < inputMatrix.Rows; i++)
                {
                    for (int j = 0; j < inputMatrix.Cols; j++)
                    {
                        dInput[matrixIndex][i][j] = dOutput[i][currentCol + j];
                    }
                }

                currentCol += inputMatrix.Cols;
            }

            return new BackwardResultBuilder()
                .AddInputGradientArray(dInput)
                .Build();
        }
    }
}
