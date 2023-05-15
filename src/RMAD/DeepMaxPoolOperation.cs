//------------------------------------------------------------------------------
// <copyright file="DeepMaxPoolOperation.cs" author="ameritusweb" date="5/14/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Max Pooling operation for a DeepMatrix.
    /// </summary>
    public class DeepMaxPoolOperation : DeepOperation
    {
        private readonly int poolSize;
        private DeepMatrix input;

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepMaxPoolOperation"/> class.
        /// </summary>
        /// <param name="poolSize">The size of the max pooling window.</param>
        public DeepMaxPoolOperation(int poolSize)
        {
            this.poolSize = poolSize;
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepMaxPoolOperation(net.Parameters.PoolSize);
        }

        /// <summary>
        /// The forward pass of the max pooling operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.input = input;

            int depth = input.Depth;
            int numRows = input.Rows;
            int numCols = input.Cols;

            // Ensure the input dimensions are divisible by the pool size
            if (numRows % this.poolSize != 0 || numCols % this.poolSize != 0)
            {
                throw new ArgumentException("Input dimensions must be divisible by the pool size.");
            }

            int pooledRows = numRows / this.poolSize;
            int pooledCols = numCols / this.poolSize;

            this.DeepOutput = new DeepMatrix(depth, pooledRows, pooledCols);

            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < pooledRows; i++)
                {
                    for (int j = 0; j < pooledCols; j++)
                    {
                        double maxVal = double.NegativeInfinity;

                        // Find the max value in the pooling window
                        for (int x = 0; x < this.poolSize; x++)
                        {
                            for (int y = 0; y < this.poolSize; y++)
                            {
                                double val = input[d, (i * this.poolSize) + x, (j * this.poolSize) + y];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                }
                            }
                        }

                        this.DeepOutput[d, i, j] = maxVal;
                    }
                }
            });

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(DeepMatrix dOutput)
        {
            int depth = this.input.Depth;
            int numRows = this.input.Rows;
            int numCols = this.input.Cols;

            DeepMatrix dInput = new DeepMatrix(depth, numRows, numCols);

            int pooledRows = numRows / this.poolSize;
            int pooledCols = numCols / this.poolSize;

            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < pooledRows; i++)
                {
                    for (int j = 0; j < pooledCols; j++)
                    {
                        // Find the max value and its index in the pooling window
                        double maxVal = double.NegativeInfinity;
                        int maxIdxX = -1, maxIdxY = -1;

                        for (int x = 0; x < this.poolSize; x++)
                        {
                            for (int y = 0; y < this.poolSize; y++)
                            {
                                double val = this.input[d, (i * this.poolSize) + x, (j * this.poolSize) + y];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdxX = x;
                                    maxIdxY = y;
                                }
                            }
                        }

                        // The gradient is passed to the input that had the maximum value
                        // Other inputs get a gradient of zero
                        for (int x = 0; x < this.poolSize; x++)
                        {
                            for (int y = 0; y < this.poolSize; y++)
                            {
                                if (x == maxIdxX && y == maxIdxY)
                                {
                                    dInput[d, (i * this.poolSize) + x, (j * this.poolSize) + y] = dOutput[d, i, j];
                                }
                                else
                                {
                                    dInput[d, (i * this.poolSize) + x, (j * this.poolSize) + y] = 0.0;
                                }
                            }
                        }
                    }
                }
            });

            return new BackwardResult() { DeepInputGradient = dInput };
        }
    }
}
