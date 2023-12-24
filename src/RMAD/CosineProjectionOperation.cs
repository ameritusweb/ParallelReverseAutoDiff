//------------------------------------------------------------------------------
// <copyright file="CosineProjectionOperation.cs" author="ameritusweb" date="12/4/2023">
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
            this.Output = new Matrix(inputMatrix.Cols, N);

            for (int j = 0; j < inputMatrix.Cols; ++j)
            {
                double decimalNumber = inputMatrix[0][j];
                double[] vector = new double[N];

                for (int i = 0; i < N; i++)
                {
                    // Calculate the influence of the i-th digit
                    double influence = this.CalculateDigitInfluence(decimalNumber, i);

                    // Generate the scaled unit vector. The scaling is based on learned parameters and the digit's influence
                    double[] scaledUnitVector = new double[N];
                    scaledUnitVector[i] = influence * Math.Pow(10, p[0][i]) / 10;

                    // Compute the magnitude of the output vector. The magnitude is adjusted based on the digit's influence
                    vector[i] = (influence / 10) * Math.Sqrt(N) / scaledUnitVector[i];
                }

                this.Output[j] = vector;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInputMatrix = new Matrix(this.inputMatrix.Rows, this.inputMatrix.Cols); // Can be left empty
            Matrix dLearnedScalings = new Matrix(this.learnedScalings.Rows, this.learnedScalings.Cols);

            for (int j = 0; j < this.inputMatrix.Cols; ++j)
            {
                double decimalNumber = this.inputMatrix[0][j];

                for (int i = 0; i < N; i++)
                {
                    // Gradient of the influence function with respect to the decimalNumber
                    double influence = this.CalculateDigitInfluence(decimalNumber, i);
                    double influenceGradient = influence * (1 - influence); // derivative of sigmoid

                    // Calculate the influence of the i-th digit on the output
                    double influenceOnOutput = dOutput[j][i] / (influence / 10) * Math.Sqrt(N);

                    // Gradient of the output with respect to the scaling factor
                    double scalingFactor = Math.Pow(10, this.learnedScalings[0][i]) / 10;
                    double scalingFactorGradient = -influenceOnOutput / (scalingFactor * scalingFactor);

                    // Adjust the gradient by the steepness of the sigmoid and the derivative of the digit influence calculation
                    dLearnedScalings[0][i] += scalingFactorGradient * Math.Log(10) * decimalNumber * Math.Pow(10, i) * influenceGradient;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputMatrix) // Since this is not needed, it's left empty
                .AddInputGradient(dLearnedScalings)
                .Build();
        }

        private double CalculateDigitInfluence(double decimalNumber, int digitPosition)
        {
            // Shift the decimal number to focus on the current digit
            double shiftedNumber = (decimalNumber * Math.Pow(10, digitPosition)) % 10;

            // Use a sigmoid function to calculate the influence
            // The sigmoid is centered around the middle of the digit (i.e., 5)
            // Adjust the steepness of the sigmoid as needed
            double steepness = 10; // Example value, adjust based on desired smoothness
            return 1 / (1 + Math.Exp(-steepness * (shiftedNumber - 5)));
        }
    }
}
