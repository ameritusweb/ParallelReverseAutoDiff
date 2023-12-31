//------------------------------------------------------------------------------
// <copyright file="MultiQuerySelfAttentionOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A multi-query self attention operation.
    /// </summary>
    public class MultiQuerySelfAttentionOperation : Operation
    {
        private Matrix inputMatrix; // NxM
        private DeepMatrix matrices; // MxM matrices
        private Matrix matrixE; // MxN

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static MultiQuerySelfAttentionOperation Instantiate(NeuralNetwork net)
        {
            return new MultiQuerySelfAttentionOperation();
        }

        /// <summary>
        /// The forward pass of the multi-query self attention operation.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="matrices">The transformed query matrices.</param>
        /// <param name="transformedTransposedKeys">The transformed transposed keys.</param>
        /// <returns>The attention scores.</returns>
        public Matrix Forward(Matrix input, DeepMatrix matrices, Matrix transformedTransposedKeys)
        {
            this.inputMatrix = input;
            this.matrices = matrices;
            this.matrixE = transformedTransposedKeys;
            int n = this.inputMatrix.Rows;
            int m = this.inputMatrix.Cols;
            Matrix[] intermediateResults = new Matrix[m];

            // Compute intermediate results
            for (int i = 0; i < m; i++)
            {
                intermediateResults[i] = this.inputMatrix * this.matrices[i]; // performs matrix multiplication
            }

            // Prepare the output matrix
            this.Output = new Matrix(n, m);

            for (int row = 0; row < n; row++)
            {
                for (int col = 0; col < m; col++)
                {
                    var intermediateRow = new Matrix(intermediateResults[col][row]);
                    var oneByOneMatrix = intermediateRow * this.matrixE.ColumnSlice(col); // performs matrix multiplication
                    this.Output[row, col] = oneByOneMatrix[0][0];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int n = this.inputMatrix.Rows;
            int m = this.inputMatrix.Cols;

            Matrix dInput = new Matrix(n, m);
            DeepMatrix dMatrices = new DeepMatrix(m, m, m);
            Matrix dTransformedTransposedKeys = new Matrix(m, n);

            // Compute gradients for each part
            for (int row = 0; row < n; row++)
            {
                for (int col = 0; col < m; col++)
                {
                    var gradient = dOutput[row, col];

                    // Gradient for dInput
                    for (int k = 0; k < m; k++)
                    {
                        dInput[row, k] += gradient * this.matrices[col][k, col] * this.matrixE[col, k];
                    }

                    // Gradient for dMatrices
                    for (int matrixIndex = 0; matrixIndex < m; matrixIndex++)
                    {
                        for (int i = 0; i < m; i++)
                        {
                            for (int j = 0; j < m; j++)
                            {
                                dMatrices[matrixIndex][i, j] += gradient * this.inputMatrix[row, i] * this.matrixE[col, j];
                            }
                        }
                    }
                }
            }

            // Gradient for dTransformedTransposedKeys
            for (int col = 0; col < m; col++)
            {
                for (int row = 0; row < n; row++)
                {
                    var gradient = dOutput[row, col];

                    // Update gradient for each element in dTransformedTransposedKeys
                    for (int k = 0; k < m; k++)
                    {
                        dTransformedTransposedKeys[k, row] += gradient * this.inputMatrix[row, k] * this.matrices[col][k, col];
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddDeepInputGradient(dMatrices)
                .AddWeightGradient(dTransformedTransposedKeys)
                .Build();
        }
    }
}
