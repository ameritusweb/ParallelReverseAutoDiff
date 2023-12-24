//------------------------------------------------------------------------------
// <copyright file="CosineProjectionOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Custom cosine projection operation.
    /// </summary>
    public class CosineProjectionOperation : Operation
    {
        private const int N = 3;
        private Matrix inputMatrix;
        private Matrix learnedScalings;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new CosineProjectionOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            // Store the intermediate matrices
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.inputMatrix, this.learnedScalings }, (x, y) => new[] { this.inputMatrix, this.learnedScalings });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            // Restore the intermediate matrices
            var restored = this.IntermediateMatrixArrays[id];
            this.inputMatrix = restored[0];
            this.learnedScalings = restored[1];
        }

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="inputMatrix">1xP input matrix.</param>
        /// <param name="p">1xN learned scalings.</param>
        /// <returns>Px(N+1) matrix.</returns>
        public Matrix Forward(Matrix inputMatrix, Matrix p)
        {
            this.inputMatrix = inputMatrix;
            this.learnedScalings = p;
            this.Output = new Matrix(inputMatrix.Cols, N + 1);

            for (int j = 0; j < inputMatrix.Cols; ++j)
            {
                double decimalNumber = inputMatrix[0][j];
                double[] vector = new double[N + 1];

                for (int i = 0; i < N; i++)
                {
                    // Scale up, extract the digit, and scale down
                    double scaledNumber = decimalNumber * Math.Pow(10, i);
                    double digit = (Math.Floor(scaledNumber) % 10) + 1; // adds 1 to prevent division by zero in the backward function

                    // Generate the scaled unit vector
                    double scaledValue = digit * Math.Pow(10, p[0][i]) / 10;

                    // Compute the magnitude of the output vector
                    vector[i + 1] = digit / 10 * Math.Sqrt(N + 1) / scaledValue;
                }

                this.Output[j] = vector;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInputMatrix = new Matrix(this.inputMatrix.Rows, this.inputMatrix.Cols); // Empty, as this is used at the beginning of the network.
            Matrix dLearnedScalings = new Matrix(this.learnedScalings.Rows, this.learnedScalings.Cols);

            for (int j = 0; j < this.inputMatrix.Cols; ++j)
            {
                double decimalNumber = this.inputMatrix[0][j];

                for (int i = 0; i < N; i++)
                {
                    double scaledNumber = decimalNumber * Math.Pow(10, i);
                    double digit = (Math.Floor(scaledNumber) % 10) + 1; // Same as in Forward

                    double scaledValue = digit * Math.Pow(10, this.learnedScalings[0][i]) / 10;
                    double magnitude = digit / 10 * Math.Sqrt(N + 1) / scaledValue;

                    // Compute the gradient with respect to the learned scaling
                    double gradientMagnitude = dOutput[j][i + 1] / (magnitude * scaledValue); // Chain rule
                    double gradientScaling = gradientMagnitude * Math.Log(10) * scaledNumber / 10;

                    dLearnedScalings[0][i] += gradientScaling;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputMatrix)
                .AddInputGradient(dLearnedScalings)
                .Build();
        }
    }
}
