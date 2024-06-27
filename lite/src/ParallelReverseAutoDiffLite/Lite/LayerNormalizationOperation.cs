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
        private const float EPSILON = 1E-9f;
        private Matrix input;
        private float[] mean;
        private float[] stdDev;
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

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateObjectArrays.AddOrUpdate(id, new[] { (object)this.input, (object)this.mean, (object)this.stdDev, (object)this.numRows, (object)this.numCols }, (x, y) => new[] { (object)this.input, (object)this.mean, (object)this.stdDev, (object)this.numRows, (object)this.numCols });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.input = (Matrix)restored[0];
            this.mean = (float[])restored[1];
            this.stdDev = (float[])restored[2];
            this.numRows = (int)restored[3];
            this.numCols = (int)restored[4];
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

            this.mean = new float[this.numRows];
            this.stdDev = new float[this.numRows];

            // Compute the mean and standard deviation for each row
            for (int i = 0; i < this.numRows; i++)
            {
                this.mean[i] = input[i].Average();
                this.stdDev[i] = PradMath.Sqrt(input[i].Select(x => PradMath.Pow(x - this.mean[i], 2)).Sum() / this.numCols);
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
                float invStdDev = 1 / (this.stdDev[i] + EPSILON);
                float exp1 = (1 - (1.0f / this.numCols)) * invStdDev;

                for (int j = 0; j < this.numCols; j++)
                {
                    var exp2 = PradMath.Pow(this.input[i][j] - this.mean[i], 2) / (this.numCols * PradMath.Pow(this.stdDev[i] + EPSILON, 3));
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