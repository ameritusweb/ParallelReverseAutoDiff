//------------------------------------------------------------------------------
// <copyright file="MatrixAddOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public class MatrixAddOperation : Operation
    {
        private double[][] inputA;
        private double[][] inputB;

        public MatrixAddOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAddOperation();
        }

        public double[][] Forward(double[][] inputA, double[][] inputB)
        {
            this.inputA = inputA;
            this.inputB = inputB;
            int numRows = inputA.Length;
            int numCols = inputA[0].Length;
            this.output = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                this.output[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    this.output[i][j] = inputA[i][j] + inputB[i][j];
                }
            }

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;
            double[][] dInputA = new double[numRows][];
            double[][] dInputB = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                dInputA[i] = new double[numCols];
                dInputB[i] = new double[numCols];
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
