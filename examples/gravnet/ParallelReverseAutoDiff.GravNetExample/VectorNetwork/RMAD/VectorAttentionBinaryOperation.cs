//------------------------------------------------------------------------------
// <copyright file="VectorAttentionBinaryOperation.cs" author="ameritusweb" date="5/2/2023">
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
    public class VectorAttentionBinaryOperation : Operation
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
            return new VectorAttentionBinaryOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector attention function.
        /// </summary>
        /// <param name="vectors">The first input to the vector attention operation.</param>
        /// <param name="probabilities">The second input to the vector attention operation.</param>
        /// <returns>The output of the vector attention operation.</returns>
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
                    // Calculate the scaling factor for the magnitude
                    double prob = probabilities[i, j];
                    double magnitudeScale = 1.5 - prob; // Ranges from 1 (at prob = 0.5) to 2 (at prob = 0) to 0.5 (at prob = 1)
                    double magnitude = vectors[i, j] * magnitudeScale;

                    // Adjust the angle based on the probability
                    double angle = vectors[i, j + M];
                    double angleAdjustment = Math.PI * (1 - prob); // Ranges from 0 (at prob = 1) to π (at prob = 0)
                    angle = (angle + angleAdjustment) % (2 * Math.PI);

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
                    double prop1 = probabilities[i, j];
                    // Gradient for magnitude
                    dVectors[i, j] = dOutput[i, j] * (1.5d - prop1); // Assuming direct gradient flow for magnitude

                    // Gradient for angle
                    double dAngle = dOutput[i, j + M];
                    dVectors[i, j + M] = dAngle; // Assuming direct gradient flow for angle

                    double dAngle_dProb1 = -Math.PI;
                    double dAngle_dProb2 = Math.PI;

                    // Gradient for Prob1 (affects magnitude and angle)
                    dProbabilities[i, j] = dAngle * dAngle_dProb1; // From derivative dAngle/dProb1 = -π

                    // Gradient for Prob2 (affects only angle)
                    dProbabilities[i, j + M] = dAngle * dAngle_dProb2; // From derivative dAngle/dProb2 = π

                    double dMagnitude = dOutput[i, j];
                    double originalMagnitude = this.vectors[i, j];

                    // Compute gradients for probabilities related to magnitude
                    dProbabilities[i, j] += -dMagnitude * originalMagnitude; // dMagnitude/dProb1
                    dProbabilities[i, j + M] += dMagnitude * originalMagnitude; // dMagnitude/dProb2
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dVectors)
                .AddInputGradient(dProbabilities)
                .Build();
        }
    }
}
