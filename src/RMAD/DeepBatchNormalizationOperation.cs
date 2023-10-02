//------------------------------------------------------------------------------
// <copyright file="DeepBatchNormalizationOperation.cs" author="ameritusweb" date="5/14/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Deep batch normalization operation.
    /// </summary>
    public class DeepBatchNormalizationOperation : DeepOperation
    {
        private const double EPSILON = 1E-6;
        private DeepMatrix input;
        private double[] means;
        private double[] vars;
        private double n;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepBatchNormalizationOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            var data = new[] { (object)this.input, (object)this.means, (object)this.vars, (object)this.n };
            this.IntermediateObjectArrays.AddOrUpdate(id, data, (_, _) => data);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.input = (DeepMatrix)restored[0];
            this.means = (double[])restored[1];
            this.vars = (double[])restored[2];
            this.n = (double)restored[3];
        }

        /// <summary>
        /// The forward pass of the deep batch normalization operation.
        /// </summary>
        /// <param name="input">The input for the deep batch normalization operation.</param>
        /// <returns>The output for the deep batch normalization operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.input = input;
            int depth = input.Depth;
            int height = input.Rows;
            int width = input.Cols;

            this.n = height * width;

            this.means = new double[depth];
            this.vars = new double[depth];

            this.DeepOutput = new DeepMatrix(depth, height, width);

            for (int d = 0; d < depth; d++)
            {
                double mean = 0;
                double var = 0;

                // Calculate mean
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        mean += input[d, i, j];
                    }
                }

                mean /= this.n;
                this.means[d] = mean;

                // Calculate variance
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        var += Math.Pow(input[d, i, j] - mean, 2);
                    }
                }

                var /= this.n;
                this.vars[d] = var;

                // Normalize
                Parallel.For(0, height, i =>
                {
                    for (int j = 0; j < width; j++)
                    {
                        this.DeepOutput[d, i, j] = (input[d, i, j] - mean) / Math.Sqrt(var + EPSILON);
                    }
                });
            }

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(DeepMatrix dOutput)
        {
            int depth = this.input.Depth;
            int height = this.input.Rows;
            int width = this.input.Cols;

            DeepMatrix dInput = new DeepMatrix(depth, height, width);

            Parallel.For(0, depth, d =>
            {
                double mean = this.means[d];
                double var = this.vars[d];
                double invStdDev = 1 / Math.Sqrt(var + EPSILON);

                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        double xMinusMean = this.input[d, i, j] - mean;

                        double dy_dvar = -0.5 * xMinusMean * Math.Pow(invStdDev, 3);
                        double dy_dmean = -invStdDev;

                        double dvar_dx = 2 * xMinusMean / this.n;

                        // Multiply local gradient by upstream gradient
                        dInput[d, i, j] = (invStdDev + (dy_dvar * dvar_dx) + dy_dmean) * dOutput[d, i, j];
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dInput)
                .Build();
        }
    }
}