//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorConstituentMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise multiplication operation.
    /// </summary>
    public class ElementwiseVectorConstituentMultiplyOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorConstituentMultiplyOperation();
        }

        /// <summary>
        /// Performs the forward operation for the Hadamard product function.
        /// </summary>
        /// <param name="input1">The first input to the Hadamard product operation.</param>
        /// <param name="input2">The second input to the Hadamard product operation.</param>
        /// <returns>The output of the Hadamard product operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            var Output = new Matrix(input1.Rows,input2.Cols);

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input2.Rows / 2; j++)
                {
                    double sumX = 0.0;
                    double sumY = 0.0;

                    for (int k = 0; k < input2.Rows / 2; k++)
                    {
                        // Accessing the magnitudes and angles from the concatenated matrices
                        double magnitude = input1[i, k];
                        double angle = input1[i, k + (input1.Cols / 2)];

                        double wMagnitude = input2[k, j];
                        double wAngle = input2[k, j + (input2.Cols / 2)];

                        // Compute vector components
                        double x1 = magnitude * Math.Cos(angle);
                        double y1 = magnitude * Math.Sin(angle);
                        double x2 = wMagnitude * Math.Cos(wAngle);
                        double y2 = wMagnitude * Math.Sin(wAngle);

                        // Select vector direction based on weight
                        double deltax = weights[k, j] > 0 ? x2 - x1 : x1 - x2;
                        double deltay = weights[k, j] > 0 ? y2 - y1 : y1 - y2;

                        // Compute resultant vector magnitude and angle
                        double resultMagnitude = Math.Sqrt((deltax * deltax) + (deltay * deltay)) * weights[k, j] * weights[k, j];
                        double resultAngle = Math.Atan2(deltay, deltax);

                        sumX += resultMagnitude * Math.Cos(resultAngle);
                        sumY += resultMagnitude * Math.Sin(resultAngle);
                    }
                    this.Output[i, j] = Math.Sqrt((sumX * sumX) + (sumY * sumY)); // Magnitude
                    this.Output[i, j + (input2.Rows / 2)] = Math.Atan2(sumY, sumX); // Angle in radians
                }
            });

            return Output;
        }

        /// <inheritdoc />
        public BackwardResult Backward(Matrix dOutput)
        {
            // Initialize gradient matrices
            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            // Loop through each element in input1
            for (int i = 0; i < input1.Rows; i++)
            {
                // Loop through each half of the columns in input1 (representing magnitudes and angles)
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Nested loop for each element in input2
                    for (int k = 0; k < input2.Rows / 2; k++)
                    {
                        double magnitude = input1[i, j];
                        double angle = input1[i, j + input1.Cols / 2];

                        double wMagnitude = input2[k, j];
                        double wAngle = input2[k, j + input2.Cols / 2];

                        // Compute vector components
                        double x1 = magnitude * Math.Cos(angle);
                        double y1 = magnitude * Math.Sin(angle);
                        double x2 = wMagnitude * Math.Cos(wAngle);
                        double y2 = wMagnitude * Math.Sin(wAngle);

                        // Delta calculations as in the forward pass
                        double deltax = weights[k, j] > 0 ? x2 - x1 : x1 - x2;
                        double deltay = weights[k, j] > 0 ? y2 - y1 : y1 - y2;

                        // Combined magnitude and angle as in the forward pass
                        double combinedMagnitude = Math.Sqrt(deltax * deltax + deltay * deltay);

                        // Calculate intermediate derivatives for combinedMagnitude and combinedAngle
                        // Derivatives for combinedMagnitude
                        double dCombinedMagnitude_dX1 = (weights[k, j] > 0 ? -1 : 1) * deltax / combinedMagnitude;
                        double dCombinedMagnitude_dY1 = (weights[k, j] > 0 ? -1 : 1) * deltay / combinedMagnitude;
                        double dCombinedMagnitude_dX2 = (weights[k, j] > 0 ? 1 : -1) * deltax / combinedMagnitude;
                        double dCombinedMagnitude_dY2 = (weights[k, j] > 0 ? 1 : -1) * deltay / combinedMagnitude;

                        // Derivatives for combinedAngle
                        double dCombinedAngle_dX1 = (weights[k, j] > 0 ? -1 : 1) * (-deltay / (deltax * deltax + deltay * deltay));
                        double dCombinedAngle_dY1 = (weights[k, j] > 0 ? -1 : 1) * (deltax / (deltax * deltax + deltay * deltay));
                        double dCombinedAngle_dX2 = (weights[k, j] > 0 ? 1 : -1) * (-deltay / (deltax * deltax + deltay * deltay));
                        double dCombinedAngle_dY2 = (weights[k, j] > 0 ? 1 : -1) * (deltax / (deltax * deltax + deltay * deltay));

                        // Apply chain rule for gradients of input1 and input2
                        dInput1[i, j] += dOutput[i, j] * dCombinedMagnitude_dX1 + dOutput[i, j + input1.Cols / 2] * dCombinedAngle_dX1;
                        dInput1[i, j + input1.Cols / 2] += dOutput[i, j] * dCombinedMagnitude_dY1 + dOutput[i, j + input1.Cols / 2] * dCombinedAngle_dY1;

                        dInput2[k, j] += dOutput[i, j] * dCombinedMagnitude_dX2 + dOutput[i, j + input2.Cols / 2] * dCombinedAngle_dX2;
                        dInput2[k, j + input2.Cols / 2] += dOutput[i, j] * dCombinedMagnitude_dY2 + dOutput[i, j + input2.Cols / 2] * dCombinedAngle_dY2;
                    }
                }
            }

            // Calculate and accumulate gradients for weights
            // (This part needs to be developed based on how weights affect the vectors in the forward pass)

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

    }
}
