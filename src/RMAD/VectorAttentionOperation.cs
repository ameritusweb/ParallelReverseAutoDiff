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
                    double prob = probabilities[i, j];
                    double magnitudeScale = 1.5 - prob; // Ranges from 1 (at prob = 0.5) to 2 (at prob = 0) to 0.5 (at prob = 1)
                    double magnitude = vectors[i, j] * magnitudeScale;

                    // Adjust the angle based on the probability
                    double angle = vectors[i, j + m];
                    double angleAdjustment = 1.5d * Math.PI * (1 - prob); // Ranges from 0 (at prob = 1) to 2π (at prob = 0)
                    angle = (angle + angleAdjustment) % (2 * Math.PI);

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
                    double prob = this.probabilities[i, j];
                    double dMagnitude_dProb = -dOutput[i, j] * this.vectors[i, j]; // Derivative of magnitude w.r.t probability
                    double dMagnitude_dProb2 = dOutput[i, j] * this.vectors[i, j]; // Derivative of magnitude w.r.t probability 2
                    double dAngle_dProb = -1.5d * Math.PI * dOutput[i, j + m]; // Derivative of angle w.r.t probability
                    double dAngle_dProb2 = 1.5d * Math.PI * dOutput[i, j + m]; // Derivative of angle w.r.t probability 2

                    // Update gradients for vectors
                    dVectors[i, j] = dOutput[i, j] * (1.5 - prob); // Corrected gradient flow for magnitude
                    dVectors[i, j + m] = dOutput[i, j + m]; // Direct gradient flow for angle is correct

                    // Aggregate gradients for probabilities
                    dProbabilities[i, j] = dMagnitude_dProb + dAngle_dProb; // Correct aggregation of gradients
                    dProbabilities[i, j + m] = dMagnitude_dProb2 + dAngle_dProb2; // Correct aggregation of gradients 2
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dVectors)
                .AddInputGradient(dProbabilities)
                .Build();
        }
    }
}
