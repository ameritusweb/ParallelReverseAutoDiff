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
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.inputMatrix, this.learnedScalings }, (key, oldValue) => new[] { this.inputMatrix, this.learnedScalings });
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
        /// <param name="inputMatrix">1xN input matrix.</param>
        /// <param name="p">1xN learned scalings.</param>
        /// <returns>Nx(N+1) matrix.</returns>
        public Matrix Forward(Matrix inputMatrix, Matrix p)
        {
            this.inputMatrix = inputMatrix;
            this.learnedScalings = p;
            this.Output = new Matrix(N, N + 1);

            for (int j = 0; j < inputMatrix.Cols; ++j)
            {
                var decimalNumber = inputMatrix[0][j];

                // Convert the decimal number to an integer representation
                int intRepresentation = (int)(decimalNumber * Math.Pow(10, N));

                for (int i = 0; i < N; i++)
                {
                    double[] vector = new double[N + 1];

                    // Add one to prevent division by zero in the backward function
                    int digit = (intRepresentation % 10) + 1;

                    // Generate the scaled unit vector. The scaling is based on learned parameters and the current digit
                    double[] scaledUnitVector = new double[N + 1];
                    scaledUnitVector[i + 1] = (double)digit * Math.Pow(10, p[0][i]) / 10;

                    // Compute the magnitude of the output vector. This magnitude is calculated in such a way that
                    // the resulting vector's cosine similarity with the scaled unit vector aligns with our desired property,
                    // which is that the cosine similarity should be equivalent to the digit divided by 10.
                    // This ensures that the vector's orientation in space (hence cosine similarity with canonical vectors)
                    // provides information about the magnitude of the digit.
                    vector[i + 1] = ((double)digit / 10) * Math.Sqrt(N + 1) / scaledUnitVector[i + 1];

                    this.Output[i] = vector;

                    intRepresentation /= 10;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInputMatrix = new Matrix(this.inputMatrix.Rows, this.inputMatrix.Cols);
            Matrix dLearnedScalings = new Matrix(this.learnedScalings.Rows, this.learnedScalings.Cols);

            for (int j = 0; j < this.inputMatrix.Cols; ++j)
            {
                // Convert the decimal number back to its integer representation
                int intRepresentation = (int)(this.inputMatrix[0][j] * Math.Pow(10, N));

                for (int i = 0; i < N; i++)
                {
                    // Extract the gradient from the output matrix for the current decimal number and digit position
                    double[] dVector = dOutput[(j * N) + i];

                    // Extract the digit and add one (this was done in the forward pass to prevent division by zero)
                    int digit = (intRepresentation % 10) + 1;
                    double scaledUnitVectorComponent = digit * Math.Pow(10, this.learnedScalings[0][i]) / 10;

                    // Compute the gradient for the scaled unit vector component based on its influence on the output and the desired cosine similarity
                    double dScaledUnitVector = dVector[i + 1] * (-digit / 10.0) * Math.Sqrt(N + 1) / Math.Pow(scaledUnitVectorComponent, 2);

                    // Compute the gradient for the scaling parameter based on its effect on the scaled unit vector and subsequently on the output
                    dLearnedScalings[0][i] += dScaledUnitVector * digit * Math.Log(10) * Math.Pow(10, this.learnedScalings[0][i]) / 10;

                    // Compute the gradient for the input decimal number by considering how changes in the digit (adjusted by adding 1) influence the output
                    double dDigit_dDecimalNumber = Math.Pow(10, N);
                    dInputMatrix[0][j] += dScaledUnitVector * 10 / digit * dDigit_dDecimalNumber;

                    intRepresentation /= 10;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputMatrix)
                .AddInputGradient(dLearnedScalings)
                .Build();
        }
    }
}
