//------------------------------------------------------------------------------
// <copyright file="MatrixAddThreeOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Matrix addition operation for three matrices or two matrices and a bias.
    /// </summary>
    public class MatrixAddThreeOperation : Operation
    {
        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAddThreeOperation();
        }

        /// <summary>
        /// Performs the forward operation for the matrix add three function.
        /// </summary>
        /// <param name="inputA">The first input to the matrix add three operation.</param>
        /// <param name="inputB">The second input to the matrix add three operation.</param>
        /// <param name="bias">The third input to the matrix add three operation.</param>
        /// <returns>The output of the matrix add operation.</returns>
        public Matrix Forward(Matrix inputA, Matrix inputB, Matrix bias)
        {
            int numRows = inputA.Length;
            int numCols = inputA[0].Length;
            this.Output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = inputA[i][j] + inputB[i][j] + bias[i][j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;
            Matrix dInputA = new Matrix(numRows, numCols);
            Matrix dInputB = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInputA[i][j] = dOutput[i][j];
                    dInputB[i][j] = dOutput[i][j];
                }
            }

            return (dInputA, dInputB); // You can return either dInputA or dInputB, as they are identical.
        }
    }
}
