//------------------------------------------------------------------------------
// <copyright file="SoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    public class SoftmaxOperation : Operation
    {
        private double[][] input;

        public SoftmaxOperation() : base()
        {
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SoftmaxOperation();
        }

        public double[][] Forward(double[][] input)
        {
            this.input = input;
            this.output = this.Softmax(input);
            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int numRows = this.output.Length;
            int numCols = this.output[0].Length;

            double[][] dLdInput = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                dLdInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    for (int k = 0; k < numCols; k++)
                    {
                        if (j == k)
                        {
                            dLdInput[i][j] += dLdOutput[i][k] * this.output[i][j] * (1 - this.output[i][j]);
                        }
                        else
                        {
                            dLdInput[i][j] -= dLdOutput[i][k] * this.output[i][j] * this.output[i][k];
                        }
                    }
                }
            }
            return (dLdInput, dLdInput);
        }

        private double[][] Softmax(double[][] input)
        {
            int numRows = input.Length;
            int numCols = input[0].Length;

            double[][] output = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                output[i] = new double[numCols];
                double max = input[i].Max();
                double sum = 0;
                for (int j = 0; j < numCols; j++)
                {
                    double exp = Math.Exp(input[i][j] - max);
                    sum += exp;
                    output[i][j] = exp;
                }

                for (int j = 0; j < numCols; j++)
                {
                    output[i][j] /= sum;
                }
            }
            return output;
        }
    }
}
