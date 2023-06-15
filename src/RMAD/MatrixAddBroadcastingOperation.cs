//------------------------------------------------------------------------------
// <copyright file="MatrixAddBroadcastingOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Matrix addition broadcasting operation.
    /// </summary>
    public class MatrixAddBroadcastingOperation : Operation
    {
        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAddBroadcastingOperation();
        }

        /// <summary>
        /// Performs the forward operation for the matrix add broadcasting function.
        /// </summary>
        /// <param name="input">The first input to the matrix add broadcasting operation.</param>
        /// <param name="bias">The bias to the matrix add broadcasting operation, a 1xN matrix where N is the number of input columns.</param>
        /// <returns>The output of the matrix add boradcasting operation.</returns>
        public Matrix Forward(Matrix input, Matrix bias)
        {
            if (input.Cols != bias.Cols)
            {
                throw new ArgumentException("The length of the bias vector must be the same as the number of columns in the input matrix.");
            }

            int numRows = input.Rows;
            int numCols = input.Cols;
            this.Output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = input[i][j] + bias[0][j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dBias = new Matrix(1, dOutput.Cols);

            // The gradient with respect to the bias is the sum of the gradients in the outputGradient
            for (int i = 0; i < dOutput.Cols; i++)
            {
                dBias[0][i] += dOutput.Column(i).Sum();
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dOutput) // return dOutput directly
                .AddBiasGradient(dBias)
                .Build();
        }
    }
}
