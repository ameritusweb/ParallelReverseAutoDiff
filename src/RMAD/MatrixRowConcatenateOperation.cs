//------------------------------------------------------------------------------
// <copyright file="MatrixRowConcatenateOperation.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A Matrix-row concatenation operation.
    /// </summary>
    public class MatrixRowConcatenateOperation : Operation
    {
        private Matrix inputMatrix;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixRowConcatenateOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.inputMatrix, this.Output }, (_, _) => new[] { this.inputMatrix, this.Output });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.inputMatrix = restored[0];
            this.Output = restored[1];
        }

        /// <summary>
        /// The forward pass of the matrix-row concatenate operation.
        /// </summary>
        /// <param name="inputMatrix">The NxM matrix.</param>
        /// <returns>The concatenated matrix.</returns>
        public Matrix Forward(Matrix inputMatrix)
        {
            this.inputMatrix = inputMatrix;
            this.Output = new Matrix(1, inputMatrix.Rows * inputMatrix.Cols);

            // Concatenate each row of the input matrix
            int idx = 0;
            for (int i = 0; i < inputMatrix.Rows; i++)
            {
                for (int j = 0; j < inputMatrix.Cols; j++)
                {
                    this.Output[0, idx++] = inputMatrix[i, j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Since the forward pass of this operation is just data reorganization,
            // the backward pass will be a reorganization of the gradients back to their original structure.
            Matrix dInputMatrix = new Matrix(this.inputMatrix.Rows, this.inputMatrix.Cols);

            int idx = 0;
            for (int i = 0; i < dInputMatrix.Rows; i++)
            {
                for (int j = 0; j < dInputMatrix.Cols; j++)
                {
                    dInputMatrix[i, j] = dOutput[0, idx++];
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputMatrix)
                .Build();
        }
    }
}
