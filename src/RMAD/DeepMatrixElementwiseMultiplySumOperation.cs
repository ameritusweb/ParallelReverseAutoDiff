//------------------------------------------------------------------------------
// <copyright file="DeepMatrixElementwiseMultiplySumOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A deep matrix element-wise multiply sum operation.
    /// </summary>
    public class DeepMatrixElementWiseMultiplySumOperation : Operation
    {
        private DeepMatrix inputMatrices; // DeepMatrix of Nx(2M) matrices
        private DeepMatrix multiplierMatrices; // DeepMatrix of 1xN matrices

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new DeepMatrixElementWiseMultiplySumOperation();
        }

        /// <summary>
        /// A forward operation.
        /// </summary>
        /// <param name="inputMatrices">The input matrices.</param>
        /// <param name="multiplierMatrices">The multiplier matrices.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix Forward(DeepMatrix inputMatrices, DeepMatrix multiplierMatrices)
        {
            this.inputMatrices = inputMatrices;
            this.multiplierMatrices = multiplierMatrices;

            int numRows = inputMatrices[0].Rows;
            int numCols = inputMatrices[0].Cols;

            // Initialize the output matrix
            this.Output = new Matrix(numRows, numCols);

            for (int i = 0; i < inputMatrices.Depth; i++)
            {
                Matrix inputMatrix = inputMatrices[i];
                Matrix multiplierMatrix = multiplierMatrices[i].Transpose(); // Transpose the 1xN matrix

                // Broadcast and element-wise multiply
                for (int row = 0; row < numRows; row++)
                {
                    for (int col = 0; col < numCols; col++)
                    {
                        this.Output[row, col] += inputMatrix[row, col] * multiplierMatrix[row, 0];
                    }
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            DeepMatrix dInputMatrices = new DeepMatrix(this.inputMatrices.Depth, this.inputMatrices.Rows, this.inputMatrices.Cols);
            DeepMatrix dMultiplierMatrices = new DeepMatrix(this.multiplierMatrices.Depth, this.multiplierMatrices.Rows, this.multiplierMatrices.Cols);

            for (int i = 0; i < this.inputMatrices.Depth; i++)
            {
                Matrix inputMatrix = this.inputMatrices[i];
                Matrix multiplierMatrix = this.multiplierMatrices[i].Transpose(); // Transposed 1xN matrix

                for (int row = 0; row < dOutput.Rows; row++)
                {
                    for (int col = 0; col < dOutput.Cols; col++)
                    {
                        // Gradient for input matrices
                        dInputMatrices[i][row, col] = dOutput[row, col] * multiplierMatrix[row, 0];

                        // Gradient for multiplier matrices, accumulating across all columns
                        dMultiplierMatrices[i][0, row] += dOutput[row, col] * inputMatrix[row, col];
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dInputMatrices)
                .AddDeepInputGradient(dMultiplierMatrices)
                .Build();
        }
    }
}
