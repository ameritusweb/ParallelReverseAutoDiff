﻿//------------------------------------------------------------------------------
// <copyright file="AddGaussianNoiseOperation.cs" author="ameritusweb" date="6/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Add Gaussian noise operation.
    /// </summary>
    public class AddGaussianNoiseOperation : Operation
    {
        private static Random rand = new Random(Guid.NewGuid().GetHashCode());
        private readonly float noiseRatio;
        private Matrix input;

        /// <summary>
        /// Initializes a new instance of the <see cref="AddGaussianNoiseOperation"/> class.
        /// </summary>
        /// <param name="noiseRatio">The noise ratio to apply.</param>
        public AddGaussianNoiseOperation(float noiseRatio)
        {
            this.noiseRatio = noiseRatio;
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new AddGaussianNoiseOperation(net.Parameters.NoiseRatio);
        }

        /// <summary>
        /// Performs the forward operation for the add Gaussian noise function.
        /// </summary>
        /// <param name="input">The matrix to add Gaussian noise to.</param>
        /// <returns>The output of the add Gaussian noise operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            float mean = input.SelectMany(x => x).Average();
            float variance = input.SelectMany(x => x.Select(val => PradMath.Pow(val - mean, 2))).Sum() / (input.Rows * input.Cols);
            float stdDev = PradMath.Sqrt(variance);

            this.Output = (Matrix)input.Clone();
            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {
                    if (rand.NextDouble() < this.noiseRatio)
                    {
                        this.Output[i][j] += this.NextGaussian() * stdDev;
                    }
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Assuming Gaussian noise is not backpropagated
            return new BackwardResultBuilder()
                .AddInputGradient(dOutput)
                .Build();
        }

        /// <summary>
        /// Returns a normally distributed random number.
        /// </summary>
        /// <returns>A normally distributed random number.</returns>
        private float NextGaussian()
        {
            float u1 = 1.0f - (float)rand.NextDouble();
            float u2 = 1.0f - (float)rand.NextDouble();
            return PradMath.Sqrt(-2.0f * PradMath.Log(u1)) * PradMath.Sin(2.0f * PradMath.PI * u2);
        }
    }
}
