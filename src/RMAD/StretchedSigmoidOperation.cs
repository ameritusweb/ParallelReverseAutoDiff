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
        private Matrix input;

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new StretchedSigmoidOperation();
        }

        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            this.output = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.output[i][j] = 1.0 / (1.0 + Math.Pow(Math.PI - 2, -input[i][j]));
                }
            }

            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dLdOutput)
        {
            int numRows = dLdOutput.Length;
            int numCols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
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
