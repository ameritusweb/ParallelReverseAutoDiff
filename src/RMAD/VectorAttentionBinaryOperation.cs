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

            int m = vectors.Cols / 2;

            Parallel.For(0, vectors.Rows, i =>
            {
                for (int j = 0; j < m; j++)
                {
                    // Calculate the scaling factor for the magnitude
                    var prob = probabilities[i, j];
                    var magnitudeScale = 1.5f - prob; // Ranges from 1 (at prob = 0.5) to 2 (at prob = 0) to 0.5 (at prob = 1)
                    var magnitude = vectors[i, j] * magnitudeScale;

                    // Adjust the angle based on the probability
                    var angle = vectors[i, j + m];
                    var angleAdjustment = PradMath.PI * (1f - prob); // Ranges from 0 (at prob = 1) to π (at prob = 0)
                    angle = (angle + angleAdjustment) % (2f * PradMath.PI);

                    this.Output[i, j] = magnitude;
                    this.Output[i, j + m] = angle;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dVectors = new Matrix(this.vectors.Rows, this.vectors.Cols);
            Matrix dProbabilities = new Matrix(this.probabilities.Rows, this.probabilities.Cols);

            int m = this.vectors.Cols / 2;
            Parallel.For(0, this.vectors.Rows, i =>
            {
                for (int j = 0; j < m; j++)
                {
                    var prop1 = this.probabilities[i, j];

                    // Gradient for magnitude
                    dVectors[i, j] = dOutput[i, j] * (1.5f - prop1); // Direct gradient flow for magnitude

                    // Gradient for angle
                    var dAngle = dOutput[i, j + m];
                    dVectors[i, j + m] = dAngle; // Direct gradient flow for angle

                    var dAngle_dProb1 = -PradMath.PI;
                    var dAngle_dProb2 = PradMath.PI;

                    // Gradient for Prob1 (affects magnitude and angle)
                    dProbabilities[i, j] = dAngle * dAngle_dProb1; // From derivative dAngle/dProb1 = -π

                    // Gradient for Prob2 (affects only angle)
                    dProbabilities[i, j + m] = dAngle * dAngle_dProb2; // From derivative dAngle/dProb2 = π

                    var dMagnitude = dOutput[i, j];
                    var originalMagnitude = this.vectors[i, j];

                    // Compute gradients for probabilities related to magnitude
                    dProbabilities[i, j] += -dMagnitude * originalMagnitude; // dMagnitude/dProb1
                    dProbabilities[i, j + m] += dMagnitude * originalMagnitude; // dMagnitude/dProb2
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dVectors)
                .AddInputGradient(dProbabilities)
                .Build();
        }
    }
}
