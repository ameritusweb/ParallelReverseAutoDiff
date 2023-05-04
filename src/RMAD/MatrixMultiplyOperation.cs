//------------------------------------------------------------------------------
// <copyright file="MatrixMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    public class MatrixMultiplyOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyOperation();
        }

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

            this.output = new Matrix(input1Rows, input2Cols);

            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                for (int j = 0; j < input2Cols; j++)
                {
                    this.output[i][j] = 0;
                    for (int k = 0; k < input1Cols; k++)
                    {
                        this.output[i][j] += input1[i][k] * input2[k][j];
                    }
                }
            });

            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
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

            return (dInput1, dInput2);
        }
    }
}
