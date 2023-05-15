//------------------------------------------------------------------------------
// <copyright file="MatrixMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Matrix multiplication operation.
    /// </summary>
    public class MatrixMultiplyOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            var op = new MatrixMultiplyOperation();
            op.HasMultipleInputs = true;
            return op;
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply function.
        /// </summary>
        /// <param name="input1">The first input to the matrix multiply operation.</param>
        /// <param name="input2">The second input to the matrix multiply operation.</param>
        /// <returns>The output of the matrix multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            this.input1 = input1;
            this.input2 = input2;
            int input1Rows = input1.Length;
            int input1Cols = input1[0].Length;
            int input2Rows = input2.Length;
            int input2Cols = input2[0].Length;

            if (input1Cols != input2Rows)
            {
                throw new InvalidOperationException("Input 1 columns do not match Input 2 rows");
            }

            this.Output = new Matrix(input1Rows, input2Cols);

            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                for (int j = 0; j < input2Cols; j++)
                {
                    this.Output[i][j] = 0;
                    for (int k = 0; k < input1Cols; k++)
                    {
                        this.Output[i][j] += input1[i][k] * input2[k][j];
                    }
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int input1Rows = this.input1.Length;
            int input1Cols = this.input1[0].Length;
            int input2Rows = this.input2.Length;
            int input2Cols = this.input2[0].Length;

            // Calculate gradient w.r.t. input1
            Matrix dInput1 = new Matrix(input1Rows, input1Cols);

            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                for (int j = 0; j < input1Cols; j++)
                {
                    dInput1[i][j] = 0;
                    for (int k = 0; k < input2Cols; k++)
                    {
                        dInput1[i][j] += dOutput[i][k] * this.input2[j][k];
                    }
                }
            });

            // Calculate gradient w.r.t. input2
            Matrix dInput2 = new Matrix(input2Rows, input2Cols);

            // Parallelize the outer loop
            Parallel.For(0, input2Rows, i =>
            {
                for (int j = 0; j < input2Cols; j++)
                {
                    dInput2[i][j] = 0;
                    for (int k = 0; k < input1Rows; k++)
                    {
                        dInput2[i][j] += this.input1[k][i] * dOutput[k][j];
                    }
                }
            });

            return new BackwardResult { InputGradientLeft = dInput1, InputGradientRight = dInput2 };
        }
    }
}
