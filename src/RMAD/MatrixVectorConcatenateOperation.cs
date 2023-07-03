//------------------------------------------------------------------------------
// <copyright file="MatrixVectorConcatenateOperation.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A Matrix-vector concatenation operation.
    /// </summary>
    public class MatrixVectorConcatenateOperation : Operation
    {
        private Matrix inputMatrix;
        private Matrix inputVector;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixVectorConcatenateOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.inputMatrix, this.inputVector, this.Output }, (key, oldValue) => new[] { this.inputMatrix, this.inputVector, this.Output });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.inputMatrix = restored[0];
            this.inputVector = restored[1];
            this.Output = restored[2];
        }

        /// <summary>
        /// The forward pass of the matrix-vector concatenate operation.
        /// </summary>
        /// <param name="inputMatrix">The NxM matrix.</param>
        /// <param name="inputVector">The 1xP vector.</param>
        /// <returns>The concatenated matrix.</returns>
        public Matrix Forward(Matrix inputMatrix, Matrix inputVector)
        {
            this.inputMatrix = inputMatrix;
            this.inputVector = inputVector;
            this.Output = new Matrix(1, (inputMatrix.Rows * inputMatrix.Cols) + inputVector.Cols);

            // Concatenate each row of the input matrix
            int idx = 0;
            for (int i = 0; i < inputMatrix.Rows; i++)
            {
                for (int j = 0; j < inputMatrix.Cols; j++)
                {
                    this.Output[0, idx++] = inputMatrix[i, j];
                }
            }

            // Concatenate the input vector
            for (int j = 0; j < inputVector.Cols; j++)
            {
                this.Output[0, idx++] = inputVector[0, j];
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Since the forward pass of this operation is just data reorganization,
            // the backward pass will be a reorganization of the gradients back to their original structure.
            Matrix dInputMatrix = new Matrix(this.inputMatrix.Rows, this.inputMatrix.Cols);
            Matrix dInputVector = new Matrix(1, this.inputVector.Cols);

            int idx = 0;
            for (int i = 0; i < dInputMatrix.Rows; i++)
            {
                for (int j = 0; j < dInputMatrix.Cols; j++)
                {
                    dInputMatrix[i, j] = dOutput[0, idx++];
                }
            }

            for (int j = 0; j < dInputVector.Cols; j++)
            {
                dInputVector[0, j] = dOutput[0, idx++];
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputMatrix)
                .AddInputGradient(dInputVector)
                .Build();
        }
    }
}
