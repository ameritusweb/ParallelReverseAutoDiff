//------------------------------------------------------------------------------
// <copyright file="MatrixHorizontalConcatenateOperation.cs" author="ameritusweb" date="12/13/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Matrix horizontal concatenation operation.
    /// </summary>
    public class MatrixHorizontalConcatenateOperation : Operation
    {
        private Matrix inputA;
        private Matrix inputB;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixHorizontalConcatenateOperation();
        }

        /// <summary>
        /// Performs the forward operation for the matrix horizontal concatenate function.
        /// </summary>
        /// <param name="inputA">The first input to the matrix horizontal concatenate operation.</param>
        /// <param name="inputB">The second input to the matrix horizontal concatenate operation.</param>
        /// <returns>The output of the matrix horizontal concatenate operation.</returns>
        public Matrix Forward(Matrix inputA, Matrix inputB)
        {
            this.inputA = inputA;
            this.inputB = inputB;

            if (inputA.Length != inputB.Length)
            {
                throw new InvalidOperationException("Matrices must have the same number of rows to concatenate horizontally.");
            }

            int numRows = inputA.Length;
            int numColsA = inputA[0].Length;
            int numColsB = inputB[0].Length;
            this.Output = new Matrix(numRows, numColsA + numColsB);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numColsA; j++)
                {
                    this.Output[i][j] = inputA[i][j];
                }

                for (int j = 0; j < numColsB; j++)
                {
                    this.Output[i][j + numColsA] = inputB[i][j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numColsA = this.inputA[0].Length; // Assuming InputA is stored from the Forward pass
            int numColsB = this.inputB[0].Length; // Assuming InputB is stored from the Forward pass

            Matrix dInputA = new Matrix(numRows, numColsA);
            Matrix dInputB = new Matrix(numRows, numColsB);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numColsA; j++)
                {
                    dInputA[i][j] = dOutput[i][j];
                }

                for (int j = 0; j < numColsB; j++)
                {
                    dInputB[i][j] = dOutput[i][j + numColsA];
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputA)
                .AddInputGradient(dInputB)
                .Build();
        }
    }
}