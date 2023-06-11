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

    /// <summary>
    /// Layer normalization operation.
    /// </summary>
    public class LayerNormalizationOperation : Operation
    {
        private const double EPSILON = 1E-9;
        private Matrix input;
        private double[] mean;
        private double[] stdDev;
        private int numRows;
        private int numCols;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new LayerNormalizationOperation();
        }

        /// <summary>
        /// The forward pass of the layer normalization operation.
        /// </summary>
        /// <param name="input">The input for the layer normalization operation.</param>
        /// <returns>The output for the layer normalization operation.</returns>
        public Matrix Forward(Matrix input)
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
            this.Output = new Matrix(this.numRows, this.numCols);

            // Parallelize the outer loop
            Parallel.For(0, this.numRows, i =>
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    this.Output[i][j] = (input[i][j] - this.mean[i]) / (this.stdDev[i] + EPSILON);
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix gradOutput)
        {
            Matrix gradient = new Matrix(this.numRows, this.numCols);

            // Parallelize the outer loop
            Parallel.For(0, this.numRows, i =>
            {
                double invStdDev = 1 / (this.stdDev[i] + EPSILON);
                double exp1 = (1 - (1.0 / this.numCols)) * invStdDev;

                for (int j = 0; j < this.numCols; j++)
                {
                    var exp2 = Math.Pow(this.input[i][j] - this.mean[i], 2) / (this.numCols * Math.Pow(this.stdDev[i] + EPSILON, 3));
                    var exp3 = exp1 - exp2;

                    // Multiply the computed gradient by the upstream gradient
                    gradient[i][j] = gradOutput[i][j] * exp3;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(gradient)
                .Build();
        }
    }
}