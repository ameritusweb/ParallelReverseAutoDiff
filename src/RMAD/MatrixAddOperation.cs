//------------------------------------------------------------------------------
// <copyright file="MatrixAddOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public class MatrixAddOperation : Operation
    {
        private Matrix inputA;
        private Matrix inputB;

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAddOperation();
        }

        public Matrix Forward(Matrix inputA, Matrix inputB)
        {
            this.inputA = inputA;
            this.inputB = inputB;
            int numRows = inputA.Length;
            int numCols = inputA[0].Length;
            this.output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.output[i][j] = inputA[i][j] + inputB[i][j];
                }
            }

            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;
            Matrix dInputA = new Matrix(numRows, numCols);
            Matrix dInputB = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInputA[i][j] = dOutput[i][j];
                    dInputB[i][j] = dOutput[i][j];
                }
            }

            return (dInputA, dInputB); // You can return either dInputA or dInputB, as they are identical.
        }
    }
}
