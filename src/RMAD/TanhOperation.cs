//------------------------------------------------------------------------------
// <copyright file="TanhOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    public class TanhOperation : Operation
    {
        private double[][] input;

        public TanhOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new TanhOperation();
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
                    this.output[i][j] = Math.Tanh(input[i][j]);
                }
            }

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;

            double[][] dInput = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                dInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    double derivative = 1 - Math.Pow(this.output[i][j], 2);
                    dInput[i][j] = dOutput[i][j] * derivative;
                }
            }

            return (dInput, dInput);
        }
    }
}
