//------------------------------------------------------------------------------
// <copyright file="BatchNormalizationOperation.cs" author="ameritusweb" date="5/12/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch normalization operation.
    /// </summary>
    public class BatchNormalizationOperation : Operation
    {
        private const double EPSILON = 1E-6;
        private Matrix input;
        private double mean;
        private double var;
        private double n;
        private int numRows;
        private int numCols;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new BatchNormalizationOperation();
        }

        /// <summary>
        /// The forward pass of the batch normalization operation.
        /// </summary>
        /// <param name="input">The input for the batch normalization operation.</param>
        /// <returns>The output for the batch normalization operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            this.numRows = input.Length;
            this.numCols = input[0].Length;

            double mean = 0;
            double var = 0;
            this.n = this.numRows * this.numCols;

            // Calculate mean
            for (int i = 0; i < this.numRows; i++)
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    mean += input[i, j];
                }
            }

            this.mean = mean / this.n;

            // Calculate variance
            for (int i = 0; i < this.numRows; i++)
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    var += Math.Pow(input[i, j] - mean, 2);
                }
            }

            this.var = var / this.n;

            // Normalize the input
            this.Output = new Matrix(this.numRows, this.numCols);

            // Parallelize the outer loop
            // Normalize activations
            Parallel.For(0, this.numRows, i =>
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    this.Output[i, j] = (input[i, j] - mean) / Math.Sqrt(var + EPSILON);
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.numRows, this.numCols);

            // Parallelize the outer loop
            Parallel.For(0, this.numRows, i =>
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    double xMinusMean = this.input[i, j] - this.mean;
                    double invStdDev = 1 / Math.Sqrt(this.var + EPSILON);

                    double dy_dvar = -0.5 * xMinusMean * Math.Pow(invStdDev, 3);
                    double dy_dmean = -invStdDev / this.n;

                    double dvar_dx = 2 * xMinusMean / this.n;

                    // Multiply local gradient by upstream gradient
                    dInput[i, j] = (invStdDev + (dy_dvar * dvar_dx) + dy_dmean) * dOutput[i, j];
                }
            });

            return (dInput, dInput);
        }
    }
}