//------------------------------------------------------------------------------
// <copyright file="MatrixTransposeOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A matrix transpose operation.
    /// </summary>
    public class MatrixTransposeOperation : Operation
    {
        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixTransposeOperation();
        }

        /// <summary>
        /// The forward pass of the matrix transpose operation.
        /// </summary>
        /// <param name="input">The input for the matrix transpose operation.</param>
        /// <returns>The output for the matrix transpose operation.</returns>
        public Matrix Forward(Matrix input)
        {
            int inputRows = input.Length;
            int inputCols = input[0].Length;

            this.Output = new Matrix(inputCols, inputRows);
            for (int i = 0; i < inputCols; i++)
            {
                for (int j = 0; j < inputRows; j++)
                {
                    this.Output[i][j] = input[j][i];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int dOutputRows = dOutput.Length;
            int dOutputCols = dOutput[0].Length;

            Matrix dInput = new Matrix(dOutputCols, dOutputRows);
            for (int i = 0; i < dOutputCols; i++)
            {
                for (int j = 0; j < dOutputRows; j++)
                {
                    dInput[i][j] = dOutput[j][i];
                }
            }

            return new BackwardResult { InputGradient = dInput };
        }
    }
}
