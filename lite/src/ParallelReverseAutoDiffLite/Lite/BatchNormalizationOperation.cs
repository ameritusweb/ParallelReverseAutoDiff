﻿//------------------------------------------------------------------------------
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
        private const float EPSILON = 1E-6F;
        private Matrix input;
        private float mean;
        private float var;
        private float n;
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

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateObjectArrays.AddOrUpdate(id, new[] { (object)this.input, (object)this.mean, (object)this.var, (object)this.n, (object)this.numRows, (object)this.numCols }, (x, y) => new[] { (object)this.input, (object)this.mean, (object)this.var, (object)this.n, (object)this.numRows, (object)this.numCols });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.input = (Matrix)restored[0];
            this.mean = (float)restored[1];
            this.var = (float)restored[2];
            this.n = (float)restored[3];
            this.numRows = (int)restored[4];
            this.numCols = (int)restored[5];
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

            this.n = this.numRows * this.numCols;

            float mean = 0;
            float var = 0;
            object meanLock = new object();
            object varLock = new object();

            Parallel.For(
                0,
                this.numRows,
                () => 0.0f,
                (i, state, localTotal) =>
                {
                    for (int j = 0; j < this.numCols; j++)
                    {
                        localTotal += input[i, j];
                    }

                    return localTotal;
                },
                (localTotal) =>
                {
                    lock (meanLock)
                    {
                        mean += localTotal;
                    }
                });

            this.mean = mean / this.n;

            Parallel.For(
                0,
                this.numRows,
                () => 0.0f,
                (i, state, localTotal) =>
                {
                    for (int j = 0; j < this.numCols; j++)
                    {
                        localTotal += PradMath.Pow(input[i, j] - mean, 2);
                    }

                    return localTotal;
                },
                (localTotal) =>
                {
                    lock (varLock)
                    {
                        var += localTotal;
                    }
                });

            this.var = var / this.n;

            // Normalize the input
            this.Output = new Matrix(this.numRows, this.numCols);

            // Parallelize the outer loop
            // Normalize activations
            Parallel.For(0, this.numRows, i =>
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    this.Output[i, j] = (input[i, j] - mean) / PradMath.Sqrt(var + EPSILON);
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.numRows, this.numCols);

            // Parallelize the outer loop
            Parallel.For(0, this.numRows, i =>
            {
                for (int j = 0; j < this.numCols; j++)
                {
                    float xMinusMean = this.input[i, j] - this.mean;
                    float invStdDev = 1 / PradMath.Sqrt(this.var + EPSILON);

                    float dy_dvar = -0.5f * xMinusMean * PradMath.Pow(invStdDev, 3);
                    float dy_dmean = -invStdDev / this.n;

                    float dvar_dx = 2 * xMinusMean / this.n;

                    // Multiply local gradient by upstream gradient
                    dInput[i, j] = (invStdDev + (dy_dvar * dvar_dx) + dy_dmean) * dOutput[i, j];
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}