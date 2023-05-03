//------------------------------------------------------------------------------
// <copyright file="MatrixMultiplyScalarOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    public class MatrixMultiplyScalarOperation : Operation
    {
        private double[][] input;
        private double scalar;

        public MatrixMultiplyScalarOperation() : base()
        {
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyScalarOperation();
        }

        public double[][] Forward(double[][] input, double scalar)
        {
            this.scalar = scalar;
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.output = new double[rows][];

            // Parallelize the outer loop
            Parallel.For(0, rows, i =>
            {
                this.output[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    this.output[i][j] = input[i][j] * this.scalar;
                }
            });

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            double[][] dLdInput = new double[rows][];

            // Parallelize the outer loop
            Parallel.For(0, rows, i =>
            {
                dLdInput[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    dLdInput[i][j] = dLdOutput[i][j] * this.scalar;
                }
            });

            return (dLdInput, dLdInput);
        }
    }

}
