//------------------------------------------------------------------------------
// <copyright file="MatrixAverageOperation.cs" author="ameritusweb" date="6/13/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Linq;

    /// <summary>
    /// Matrix average operation.
    /// </summary>
    public class MatrixAverageOperation : Operation
    {
        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAverageOperation();
        }

        /// <summary>
        /// Performs the forward operation for the matrix average function.
        /// </summary>
        /// <param name="input">The input to the matrix average operation.</param>
        /// <returns>The output of the matrix average operation.</returns>
        public Matrix Forward(Matrix input)
        {
            int numRows = input.Rows;
            this.Output = new Matrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                this.Output[i][0] = input[i].Average();
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = this.Output[0].Length;

            Matrix dInput = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInput[i][j] = dOutput[i][0] / numCols;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}
