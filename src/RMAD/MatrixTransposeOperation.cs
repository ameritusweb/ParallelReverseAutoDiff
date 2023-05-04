//------------------------------------------------------------------------------
// <copyright file="MatrixTransposeOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public class MatrixTransposeOperation : Operation
    {
        private Matrix input;

        public MatrixTransposeOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixTransposeOperation();
        }

        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int inputRows = input.Length;
            int inputCols = input[0].Length;

            this.output = new Matrix(inputCols, inputRows);
            for (int i = 0; i < inputCols; i++)
            {
                for (int j = 0; j < inputRows; j++)
                {
                    this.output[i][j] = input[j][i];
                }
            }

            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
        {
            int dOutputRows = dOutput.Length;
            int dOutputCols = dOutput[0].Length;

            Matrix dInput = new Matrix(dOutputCols, dOutputRows);
            for (int i = 0; i < dOutputCols; i++)
            {
                for (int j = 0; j < dOutputRows; j++)
                {
                    dInput[i][j] = dOutput[j][i];
                }
            }

            return (dInput, dInput);
        }
    }
}
