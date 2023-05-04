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
        private Matrix input;

        public SigmoidOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SigmoidOperation();
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
                    this.output[i][j] = 1.0 / (1.0 + Math.Exp(-input[i][j]));
                }
            }

            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
        {
            int numRows = this.input.Length;
            int numCols = this.input[0].Length;

            Matrix dInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
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
