//------------------------------------------------------------------------------
// <copyright file="StretchedSigmoidOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    public class StretchedSigmoidOperation : Operation
    {
        private double[][] input;

        public StretchedSigmoidOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new StretchedSigmoidOperation();
        }

        public double[][] Forward(double[][] input)
        {
            this.input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            this.output = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                this.output[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    this.output[i][j] = 1.0 / (1.0 + Math.Pow(Math.PI - 2, -input[i][j]));
                }
            }

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int numRows = dLdOutput.Length;
            int numCols = dLdOutput[0].Length;
            double[][] dLdInput = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                dLdInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    double x = this.input[i][j];
                    double dx = Math.Pow(Math.PI - 2, -x) * Math.Log(Math.PI - 2) / Math.Pow(1 + Math.Pow(Math.PI - 2, -x), 2);
                    dLdInput[i][j] = dLdOutput[i][j] * dx;
                }
            }

            return (dLdInput, dLdInput);
        }
    }
}
