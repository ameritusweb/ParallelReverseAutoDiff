//------------------------------------------------------------------------------
// <copyright file="VectorizeOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Performs the forward and backward operations for the vectorize function.
    /// </summary>
    public class VectorizeOperation : Operation
    {
        private Matrix input;
        private Matrix angles;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new VectorizeOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrices.AddOrUpdate(id, this.input, (x, y) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the vectorize function.
        /// </summary>
        /// <param name="input">The input to the vectorize operation.</param>
        /// <param name="angles">The angles to the vectorize operation.</param>
        /// <returns>The output of the vectorize operation.</returns>
        public Matrix Forward(Matrix input, Matrix angles)
        {
            this.input = input;
            this.angles = angles;

            int rows = input.Length;
            int cols = input[0].Length;

            if (cols != angles[0].Length)
            {
                throw new ArgumentException("Input and angles matrices must have the same number of columns.");
            }

            int m = cols * 2;
            this.Output = new Matrix(rows, m);

            var angleMappings = new Dictionary<(float, float, float), float>();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // Left half: magnitudes from input
                    this.Output[i][j] = input[i][j];

                    // Right half: angles with context
                    float prev = j > 0 ? input[i][j - 1] : float.MinValue;
                    float next = j < cols - 1 ? input[i][j + 1] : float.MaxValue;
                    var context = (prev, input[i][j], next);

                    if (!angleMappings.ContainsKey(context))
                    {
                        angleMappings[context] = angles[i][j]; // Store new angle for the context
                    }

                    this.Output[i][cols + j] = angleMappings[context]; // Use the angle for the output matrix
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = this.input[0].Length;
            Matrix dLdInput = new Matrix(rows, cols); // Gradient with respect to input magnitudes
            Matrix dLdAngles = new Matrix(rows, cols); // Gradient with respect to input angles

            var angleUsage = new Dictionary<float, List<(int, int)>>(); // Tracks angle usage

            // Track the usage of each angle in the forward pass
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float angle = this.angles[i][j];
                    if (!angleUsage.ContainsKey(angle))
                    {
                        angleUsage[angle] = new List<(int, int)>();
                    }

                    angleUsage[angle].Add((i, j));
                }
            }

            // Calculate the gradient for each angle based on its usage
            foreach (var angle in angleUsage.Keys)
            {
                foreach (var (row, col) in angleUsage[angle])
                {
                    // Accumulate gradients from all outputs affected by this angle
                    dLdAngles[row][col] += dLdOutput[row][cols + col];
                }
            }

            // Gradient for magnitudes is straightforward
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    dLdInput[i][j] = dLdOutput[i][j];
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .AddInputGradient(dLdAngles)
                .Build();
        }
    }
}
