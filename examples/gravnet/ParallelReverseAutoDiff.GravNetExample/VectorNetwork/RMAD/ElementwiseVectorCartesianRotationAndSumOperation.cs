//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCartesianRotationAndSumOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Element-wise cartesian rotation and sum operation.
    /// </summary>
    public class ElementwiseVectorCartesianRotationAndSumOperation : Operation
    {
        private Matrix rotationTargets;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorCartesianRotationAndSumOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector rotation and sum function.
        /// </summary>
        /// <param name="inputVectors">The first input to the element-wise vector rotation and sum operation.</param>
        /// <param name="rotationTargets">The second input to the element-wise vector rotation and sum operation.</param>
        /// <returns>The output of the element-wise vector rotation and sum operation.</returns>
        public Matrix Forward(Matrix inputVectors, Matrix rotationTargets)
        {
            this.rotationTargets = rotationTargets;

            double clockwise = -Math.PI / 2;

            // Pre-calculate the cosine and sine for both rotation angles
            double cosClockwise = Math.Cos(clockwise);
            double sinClockwise = Math.Sin(clockwise);

            // Initialize the summation vector
            double sumX = 0.0;
            double sumY = 0.0;

            int vectorIndex = 0;
            for (int i = 0; i < 15; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    double x = inputVectors[vectorIndex, 0];
                    double y = inputVectors[vectorIndex, 1];

                    double cosTheta = rotationTargets[i, j] == 1 ? cosClockwise : 0d;
                    double sinTheta = rotationTargets[i, j] == 1 ? sinClockwise : 0d;

                    // Apply rotation
                    double rotatedX = (x * cosTheta) - (y * sinTheta);
                    double rotatedY = (x * sinTheta) + (y * cosTheta);

                    // Summation
                    sumX += rotatedX;
                    sumY += rotatedY;

                    vectorIndex++;
                }
            }

            // Output is a 1x2 matrix of X and Y components for the output summation vector
            Matrix output = new Matrix(1, 2);
            output[0, 0] = sumX;
            output[0, 1] = sumY;
            this.Output = output;

            return this.Output;
        }


        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            double clockwise = -Math.PI / 2;

            // Initialize dInputVectors with the same shape as the forward input vectors
            Matrix dInputVectors = new Matrix(225, 2);

            int vectorIndex = 0;
            for (int i = 0; i < 15; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    var rotation = rotationTargets[i, j] == 1 ? clockwise : 0d;

                    double cosTheta = Math.Cos(rotation);
                    double sinTheta = Math.Sin(rotation);

                    dInputVectors[vectorIndex, 0] = (dOutput[0, 0] * cosTheta) + (dOutput[0, 1] * sinTheta);
                    dInputVectors[vectorIndex, 1] = (-dOutput[0, 0] * sinTheta) + (dOutput[0, 1] * cosTheta);

                    vectorIndex++;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputVectors)
                .Build();
        }
    }
}
