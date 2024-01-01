//------------------------------------------------------------------------------
// <copyright file="DeepConcatenateOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A deep concatenate operation.
    /// </summary>
    public class DeepConcatenateOperation : DeepOperation
    {
        private DeepMatrix inputMatrices; // Array of NxM matrices

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepConcatenateOperation();
        }

        /// <summary>
        /// The forward pass of the concatenate operation.
        /// </summary>
        /// <param name="inputMatrices">The input matrices.</param>
        /// <returns>The masked input.</returns>
        public DeepMatrix Forward(DeepMatrix inputMatrices)
        {
            this.inputMatrices = inputMatrices;
            int numPairs = inputMatrices.Depth * (inputMatrices.Depth - 1);
            this.DeepOutput = new DeepMatrix(numPairs, inputMatrices.Rows, inputMatrices.Cols * 2);

            int index = 0;
            for (int i = 0; i < inputMatrices.Depth; i++)
            {
                for (int j = 0; j < inputMatrices.Depth; j++)
                {
                    if (i != j)
                    {
                        this.DeepOutput[index] = inputMatrices[i].ConcatenateColumns(inputMatrices[j]);
                        index++;
                    }
                }
            }

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(DeepMatrix dConcatenatedMatrices)
        {
            // Initialize gradients for the original input matrices
            DeepMatrix dInputMatrices = new DeepMatrix(this.inputMatrices.Depth, this.inputMatrices.Rows, this.inputMatrices.Cols);

            for (int i = 0; i < dInputMatrices.Depth; i++)
            {
                dInputMatrices[i] = new Matrix(this.inputMatrices.Rows, this.inputMatrices.Cols);
            }

            int index = 0;
            for (int i = 0; i < this.inputMatrices.Depth; i++)
            {
                for (int j = 0; j < this.inputMatrices.Depth; j++)
                {
                    if (i != j)
                    {
                        Matrix dConcatenatedMatrix = dConcatenatedMatrices[index];

                        // Split the gradient matrix back into two parts
                        for (int row = 0; row < dConcatenatedMatrix.Rows; row++)
                        {
                            for (int col = 0; col < this.inputMatrices.Cols; col++)
                            {
                                dInputMatrices[i][row, col] += dConcatenatedMatrix[row, col]; // Gradient for the first matrix in the pair
                                dInputMatrices[j][row, col] += dConcatenatedMatrix[row, col + this.inputMatrices.Cols]; // Gradient for the second matrix in the pair
                            }
                        }

                        index++;
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dInputMatrices)
                .Build();
        }
    }
}
