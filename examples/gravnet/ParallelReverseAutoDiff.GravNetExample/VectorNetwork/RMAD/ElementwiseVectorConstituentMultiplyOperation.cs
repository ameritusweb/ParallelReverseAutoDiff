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
        private Matrix sumX;
        private Matrix sumY;
        private Matrix slopesX;
        private Matrix slopesY;

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

            this.Output = new Matrix(input1.Rows,input2.Cols);
            this.sumX = new Matrix(input1.Rows, input2.Cols / 2);
            this.sumY = new Matrix(input1.Rows, input2.Cols / 2);
            this.slopesX = new Matrix(input1.Rows, input2.Cols / 2);
            this.slopesY = new Matrix(input1.Rows, input2.Cols / 2);

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input2.Cols / 2; j++)
                {
                    double sumX = 0.0;
                    double sumY = 0.0;

                    double[] resultMagnitudes = new double[input2.Rows / 2];
                    double[] resultAngles = new double[input2.Rows / 2];

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
                        double resultMagnitude = Math.Sqrt((deltax * deltax) + (deltay * deltay)) * weights[k, j];
                        resultMagnitudes[k] = resultMagnitude;
                        double resultAngle = Math.Atan2(deltay, deltax);
                        resultAngles[k] = resultAngle;

                        sumX += resultMagnitude * Math.Cos(resultAngle);
                        sumY += resultMagnitude * Math.Sin(resultAngle);
                    }

                    for (int k = 0; k < input1.Cols / 2; k++)
                    {
                        double perturbedResultMagnitude = resultMagnitudes[k] * 0.0001d;
                        double rx = resultMagnitudes.Take(k).Concat(resultMagnitudes.Skip(k + 1)).Sum(x => x * Math.Cos(resultAngles[k]));
                        rx += perturbedResultMagnitude * Math.Cos(resultAngles[k]);
                        double ry = resultMagnitudes.Take(k).Concat(resultMagnitudes.Skip(k + 1)).Sum(x => x * Math.Sin(resultAngles[k]));
                        ry += perturbedResultMagnitude * Math.Sin(resultAngles[k]);

                        double resultMagnitudeChange = perturbedResultMagnitude - resultMagnitudes[k];
                        double sumXChange = rx - sumX;

                        double slopeX = sumXChange / resultMagnitudeChange;

                        double sumYChange = ry - sumY;

                        double slopeY = sumYChange / resultMagnitudeChange;

                        this.slopesX[i, j] = slopeX;
                        this.slopesY[i, j] = slopeY;
                    }

                    this.sumX[i, j] = sumX;
                    this.sumY[i, j] = sumY;

                    this.Output[i, j] = Math.Sqrt((sumX * sumX) + (sumY * sumY)); // Magnitude
                    this.Output[i, j + (input2.Rows / 2)] = Math.Atan2(sumY, sumX); // Angle in radians
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Initialize gradient matrices
            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            // Loop through each element in input1
            Parallel.For(0, input1.Rows, i =>
            {
                // Loop through each half of the columns in input1 (representing magnitudes and angles)
                for (int j = 0; j < input2.Cols / 2; j++)
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

                        double dSumX_dDeltaX = this.slopesX[i, j]; // empirically determined
                        double dSumY_dDeltaY = this.slopesY[i, j]; // empirically determined

                        // Derivatives of delta components with respect to inputs
                        double dDeltaX_dX1 = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        double dDeltaY_dY1 = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        double dDeltaX_dX2 = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign
                        double dDeltaY_dY2 = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign

                        // Analytically determined gradients for combined magnitude
                        double dCombinedMagnitude_dSumX = this.sumX[i, j] / Math.Sqrt(this.sumX[i, j] * this.sumX[i, j] + this.sumY[i, j] * this.sumY[i, j]);
                        double dCombinedMagnitude_dSumY = this.sumY[i, j] / Math.Sqrt(this.sumX[i, j] * this.sumX[i, j] + this.sumY[i, j] * this.sumY[i, j]);

                        double dCombinedAngle_dSumX = -deltay / (deltax * deltax + deltay * deltay);
                        double dCombinedAngle_dSumY = deltax / (deltax * deltax + deltay * deltay);

                        // Apply chain rule for dInput1
                        dInput1[i, j] += dOutput[i, j] * (dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dX1 + dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dX1);
                        dInput1[i, j + input1.Cols / 2] += dOutput[i, j] * (dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dY1 + dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dY1);

                        // Apply chain rule for dInput2
                        dInput2[k, j] += dOutput[i, j] * (dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dX2 + dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dX2);
                        dInput2[k, j + input2.Cols / 2] += dOutput[i, j] * (dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dY2 + dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dY2);

                        // Derivatives of delta components with respect to weight
                        double dDeltaX_dWeight = (weights[k, j] > 0) ? (x2 - x1) : (x1 - x2);
                        double dDeltaY_dWeight = (weights[k, j] > 0) ? (y2 - y1) : (y1 - y2);

                        // Apply chain rule for weights for magnitude and angle
                        dWeights[k, j] += dOutput[i, j] *
                                          (dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dWeight +
                                           dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dWeight);

                        dWeights[k, j] += dOutput[i, j + input1.Cols / 2] *
                                          (dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dWeight +
                                           dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dWeight);

                    }
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

    }
}
