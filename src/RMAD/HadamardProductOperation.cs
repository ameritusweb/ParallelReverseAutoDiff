//------------------------------------------------------------------------------
// <copyright file="HadamardProductOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    public class HadamardProductOperation : Operation
    {
        private double[][] input1;
        private double[][] input2;

        public HadamardProductOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new HadamardProductOperation();
        }

        public double[][] Forward(double[][] input1, double[][] input2)
        {
            this.input1 = input1;
            this.input2 = input2;
            int numRows = input1.Length;
            int numCols = input1[0].Length;

            this.output = new double[numRows][];

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                this.output[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    this.output[i][j] = input1[i][j] * input2[i][j];
                }
            });

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = this.input1.Length;
            int numCols = this.input1[0].Length;

            // Calculate gradient w.r.t. input1
            double[][] dInput1 = new double[numRows][];
            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                dInput1[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    dInput1[i][j] = dOutput[i][j] * this.input2[i][j];
                }
            });

            // Calculate gradient w.r.t. input2
            double[][] dInput2 = new double[numRows][];
            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                dInput2[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    dInput2[i][j] = dOutput[i][j] * this.input1[i][j];
                }
            });

            return (dInput1, dInput2);
        }
    }
}
