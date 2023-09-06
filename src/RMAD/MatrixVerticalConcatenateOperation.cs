//------------------------------------------------------------------------------
// <copyright file="MatrixVerticalConcatenateOperation.cs" author="ameritusweb" date="6/13/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Matrix vertical concatenate operation.
    /// </summary>
    public class MatrixVerticalConcatenateOperation : Operation
    {
        private DeepMatrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixVerticalConcatenateOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.input, (key, oldValue) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateDeepMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the matrix vertical concatenate function.
        /// </summary>
        /// <param name="input">A DeepMatrix containing 1xM matrices.</param>
        /// <returns>The output of the matrix vertical concatenate operation.</returns>
        public Matrix Forward(DeepMatrix input)
        {
            this.input = input;
            int totalRows = input.Depth;  // Since each matrix is 1xM
            int numCols = input[0].Cols;

            this.Output = new Matrix(totalRows, numCols);
            int currentRow = 0;

            foreach (Matrix matrix in input)
            {
                if (matrix.Rows != 1 || matrix.Cols != numCols)
                {
                    throw new ArgumentException("All input matrices must be 1xM and have the same number of columns.");
                }

                for (int j = 0; j < numCols; j++)
                {
                    this.Output[currentRow][j] = matrix[0][j];
                }

                currentRow += 1;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Rows;
            int numCols = dOutput.Cols;
            DeepMatrix dInput = new DeepMatrix(numRows, 1, numCols);  // Initialize with depth equal to numRows

            for (int i = 0; i < numRows; i++)
            {
                dInput[i] = new Matrix(1, numCols);  // Each matrix is 1xM

                for (int j = 0; j < numCols; j++)
                {
                    dInput[i][0][j] = dOutput[i][j];
                }
            }

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dInput)
                .Build();
        }
    }
}
