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
        private double[][] input1;
        private double[][] input2;

        public MatrixMultiplyOperation() : base()
        {
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyOperation();
        }

        public double[][] Forward(double[][] input1, double[][] input2)
        {
            this.input1 = input1;
            this.input2 = input2;
            int input1Rows = input1.Length;
            int input1Cols = input1[0].Length;
            int input2Rows = input2.Length;
            int input2Cols = input2[0].Length;

            if (input1Cols != input2Rows)
            {
                throw new Exception("Input 1 columns do not match Input 2 rows");
            }

            this.output = new double[input1Rows][];

            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                this.output[i] = new double[input2Cols];
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

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int input1Rows = this.input1.Length;
            int input1Cols = this.input1[0].Length;
            int input2Rows = this.input2.Length;
            int input2Cols = this.input2[0].Length;

            // Calculate gradient w.r.t. input1
            double[][] dInput1 = new double[input1Rows][];
            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                dInput1[i] = new double[input1Cols];
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
            double[][] dInput2 = new double[input2Rows][];

            // Parallelize the outer loop
            Parallel.For(0, input2Rows, i =>
            {
                dInput2[i] = new double[input2Cols];
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
