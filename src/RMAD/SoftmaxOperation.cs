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
        private Matrix input;

        public SoftmaxOperation() : base()
        {
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SoftmaxOperation();
        }

        public Matrix Forward(Matrix input)
        {
            this.input = input;
            this.output = this.Softmax(input);
            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dLdOutput)
        {
            int numRows = this.output.Length;
            int numCols = this.output[0].Length;

            Matrix dLdInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
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

        private Matrix Softmax(Matrix input)
        {
            int numRows = input.Length;
            int numCols = input[0].Length;

            Matrix output = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
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
