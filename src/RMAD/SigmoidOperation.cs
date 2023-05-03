//------------------------------------------------------------------------------
// <copyright file="SigmoidOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    public class SigmoidOperation : Operation
    {
        private double[][] input;

        public SigmoidOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SigmoidOperation();
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
                    this.output[i][j] = 1.0 / (1.0 + Math.Exp(-input[i][j]));
                }
            }

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = this.input.Length;
            int numCols = this.input[0].Length;

            double[][] dInput = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                dInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    double sigmoidDerivative = this.output[i][j] * (1 - this.output[i][j]);
                    dInput[i][j] = dOutput[i][j] * sigmoidDerivative;
                }
            }

            return (dInput, dInput);
        }
    }
}
