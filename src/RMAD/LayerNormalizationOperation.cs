//------------------------------------------------------------------------------
// <copyright file="LayerNormalizationOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    public class LayerNormalizationOperation : Operation
    {

        private const double epsilon = 1E-6;
        private double[][] input;
        private double[] mean;
        private double[] stdDev;
        private int numRows;
        private int numCols;

        public LayerNormalizationOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new LayerNormalizationOperation();
        }

        public double[][] Forward(double[][] input)
        {
            this.input = input;
            this.numRows = input.Length;
            this.numCols = input[0].Length;

            this.mean = new double[this.numRows];
            this.stdDev = new double[this.numRows];

            // Compute the mean and standard deviation for each row
            for (int i = 0; i < this.numRows; i++)
            {
                this.mean[i] = input[i].Average();
                this.stdDev[i] = Math.Sqrt(input[i].Select(x => Math.Pow(x - this.mean[i], 2)).Sum() / this.numCols);
            }

            // Normalize the input
            this.output = new double[this.numRows][];

            // Parallelize the outer loop
            Parallel.For(0, this.numRows, i =>
            {
                this.output[i] = new double[this.numCols];
                for (int j = 0; j < this.numCols; j++)
                {
                    this.output[i][j] = (input[i][j] - this.mean[i]) / (this.stdDev[i] + epsilon);
                }
            });

            return this.output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] gradOutput)
        {
            double[][] gradient = new double[this.numRows][];

            // Parallelize the outer loop
            Parallel.For(0, this.numRows, i =>
            {
                gradient[i] = new double[this.numCols];
                double invStdDev = 1 / (this.stdDev[i] + epsilon);
                double exp1 = (1 - (1.0 / this.numCols)) * invStdDev;

                for (int j = 0; j < this.numCols; j++)
                {
                    var exp2 = Math.Sqrt(Math.Pow(this.input[i][j] - this.mean[i], 2)) / (this.numCols * Math.Pow(this.stdDev[i] + epsilon, 2));
                    var exp3 = exp1 - exp2;
                    // Multiply the computed gradient by the upstream gradient
                    gradient[i][j] = gradOutput[i][j] * exp3;
                }
            });

            return (gradient, gradient);
        }
    }
}