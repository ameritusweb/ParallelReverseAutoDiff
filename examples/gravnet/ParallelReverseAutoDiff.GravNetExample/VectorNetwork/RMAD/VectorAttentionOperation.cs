//------------------------------------------------------------------------------
// <copyright file="VectorAttentionOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Vector attention operation.
    /// </summary>
    public class VectorAttentionOperation : Operation
    {
        private Matrix vectors;
        private Matrix probabilities;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new VectorAttentionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector attention function.
        /// </summary>
        /// <param name="vectors">The first input to the vector attention operation.</param>
        /// <param name="probabilities">The second input to the vector attention operation.</param>
        /// <returns>The output of the vector attentionn operation.</returns>
        public Matrix Forward(Matrix vectors, Matrix probabilities)
        {
            this.vectors = vectors;
            this.probabilities = probabilities;

            this.Output = new Matrix(vectors.Rows, vectors.Cols);

            int M = vectors.Cols / 2;

            Parallel.For(0, vectors.Rows, i =>
            {
                for (int j = 0; j < M; j++)
                {
                    // Check the existence probability
                    double magnitude = probabilities[i, j] > 0.5 ? vectors[i, j] : 0;

                    // Check the angle probability and calculate the opposite angle if needed
                    double angle = vectors[i, j + M];
                    if (probabilities[i, j + M] > 0.5)
                    {
                        angle = (angle + Math.PI) % (2 * Math.PI); // Opposite angle
                    }

                    this.Output[i, j] = magnitude;
                    this.Output[i, j + M] = angle;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dVectors = new Matrix(vectors.Rows, vectors.Cols);
            Matrix dProbabilities = new Matrix(probabilities.Rows, probabilities.Cols);

            int M = vectors.Cols / 2;
            Parallel.For(0, vectors.Rows, i =>
            {
                for (int j = 0; j < M; j++)
                {
                    // Gradient for magnitude
                    dVectors[i, j] = probabilities[i, j] > 0.5 ? dOutput[i, j] : 0;
                    dProbabilities[i, j] = (probabilities[i, j] > 0.5 ? 1 : 0) * dOutput[i, j];

                    // Gradient for angle
                    bool isFlipped = probabilities[i, j + M] > 0.5;
                    double dAngle = dOutput[i, j + M];
                    dVectors[i, j + M] = isFlipped ? -dAngle : dAngle;

                    // Consistent gradient for probability w.r.t angle
                    // The gradient indicates the direction to adjust the probability
                    dProbabilities[i, j + M] = dAngle * (isFlipped ? -1 : 1);
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dVectors)
                .AddInputGradient(dProbabilities)
                .Build();
        }
    }
}
