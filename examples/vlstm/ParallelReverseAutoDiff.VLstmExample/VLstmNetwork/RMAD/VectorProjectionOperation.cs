//------------------------------------------------------------------------------
// <copyright file="VectorProjectionOperation.cs" author="ameritusweb" date="1/21/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Vector projection operation.
    /// </summary>
    public class VectorProjectionOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private Matrix sumX;
        private Matrix sumY;
        private Matrix dSumXDDeltaX;
        private Matrix dSumXDDeltaY;
        private Matrix dSumYDDeltaX;
        private Matrix dSumYDDeltaY;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new VectorProjectionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector projection function.
        /// </summary>
        /// <param name="input1">The first input to the vector projection operation.</param>
        /// <param name="input2">The second input to the vector projection operation.</param>
        /// <returns>The output of the vector projection operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(1,input2.Cols);
            this.sumX = new Matrix(1, input2.Cols / 2);
            this.sumY = new Matrix(1, input2.Cols / 2);

            this.dSumXDDeltaX = new Matrix(1, input2.Cols / 2);
            this.dSumXDDeltaY = new Matrix(1, input2.Cols / 2);
            this.dSumYDDeltaX = new Matrix(1, input2.Cols / 2);
            this.dSumYDDeltaY = new Matrix(1, input2.Cols / 2);

            Parallel.For(0, input2.Cols / 2, j =>
            {
                double sumX = 0.0;
                double sumY = 0.0;

                double dSumX_dDeltaX = 0.0;
                double dSumX_dDeltaY = 0.0;
                double dSumY_dDeltaX = 0.0;
                double dSumY_dDeltaY = 0.0;

                for (int k = 0; k < input2.Rows / 2; k++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude = input1[0, 0];
                    double angle = input1[0, 1];

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
                    double resultAngle = Math.Atan2(deltay, deltax);

                    double dResultMagnitude_dDeltaX = (deltax * weights[k, j]) / Math.Sqrt(deltax * deltax + deltay * deltay);
                    double dResultMagnitude_dDeltaY = (deltay * weights[k, j]) / Math.Sqrt(deltax * deltax + deltay * deltay);
                    double dResultAngle_dDeltaX = -deltay / (deltax * deltax + deltay * deltay);
                    double dResultAngle_dDeltaY = deltax / (deltax * deltax + deltay * deltay);

                    double localSumX = resultMagnitude * Math.Cos(resultAngle);
                    double localSumY = resultMagnitude * Math.Sin(resultAngle);

                    double dLocalSumX_dResultMagnitude = Math.Cos(resultAngle);
                    double dLocalSumY_dResultMagnitude = Math.Sin(resultAngle);

                    double dLocalSumX_dResultAngle = -resultMagnitude * Math.Sin(resultAngle);
                    double dLocalSumY_dResultAngle = resultMagnitude * Math.Cos(resultAngle);

                    double dLocalSumX_dDeltaX = dLocalSumX_dResultMagnitude * dResultMagnitude_dDeltaX
                        + dLocalSumX_dResultAngle * dResultAngle_dDeltaX;
                    double dLocalSumX_dDeltaY = dLocalSumX_dResultMagnitude * dResultMagnitude_dDeltaY
                        + dLocalSumX_dResultAngle * dResultAngle_dDeltaY;
                    double dLocalSumY_dDeltaX = dLocalSumY_dResultMagnitude * dResultMagnitude_dDeltaX
                        + dLocalSumY_dResultAngle * dResultAngle_dDeltaX;
                    double dLocalSumY_dDeltaY = dLocalSumY_dResultMagnitude * dResultMagnitude_dDeltaY
                        + dLocalSumY_dResultAngle * dResultAngle_dDeltaY;

                    sumX += localSumX;
                    sumY += localSumY;

                    dSumX_dDeltaX += dLocalSumX_dDeltaX;
                    dSumX_dDeltaY += dLocalSumX_dDeltaY;
                    dSumY_dDeltaX += dLocalSumY_dDeltaX;
                    dSumY_dDeltaY += dLocalSumY_dDeltaY;
                }

                this.sumX[0, j] = sumX;
                this.sumY[0, j] = sumY;

                this.dSumXDDeltaX[0, j] = dSumX_dDeltaX;
                this.dSumXDDeltaY[0, j] = dSumX_dDeltaY;
                this.dSumYDDeltaX[0, j] = dSumY_dDeltaX;
                this.dSumYDDeltaY[0, j] = dSumY_dDeltaY;

                this.Output[0, j] = Math.Sqrt((sumX * sumX) + (sumY * sumY)); // Magnitude
                this.Output[0, j + (input2.Rows / 2)] = Math.Atan2(sumY, sumX); // Angle in radians
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Initialize gradient matrices
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            // Loop through each half of the columns in input1 (representing magnitudes and angles)
            Parallel.For(0, this.input2.Cols / 2, j =>
            {
                // Nested loop for each element in input2
                for (int k = 0; k < this.input2.Rows / 2; k++)
                {
                    double magnitude = this.input1[0, j];
                    double angle = this.input1[0, j + this.input1.Cols / 2];

                    double wMagnitude = this.input2[k, j];
                    double wAngle = this.input2[k, j + this.input2.Cols / 2];

                    // Compute vector components
                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    // Delta calculations as in the forward pass
                    double deltax = this.weights[k, j] > 0 ? x2 - x1 : x1 - x2;
                    double deltay = this.weights[k, j] > 0 ? y2 - y1 : y1 - y2;

                    double dSumX_dDeltaX = this.dSumXDDeltaX[0, j];
                    double dSumX_dDeltaY = this.dSumXDDeltaY[0, j];
                    double dSumY_dDeltaX = this.dSumYDDeltaX[0, j];
                    double dSumY_dDeltaY = this.dSumYDDeltaY[0, j];

                    // Derivatives of delta components with respect to inputs
                    double dDeltaX_dX1 = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                    double dDeltaY_dY1 = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                    double dDeltaX_dX2 = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign
                    double dDeltaY_dY2 = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign

                    // Analytically determined gradients for combined magnitude
                    double dCombinedMagnitude_dSumX = this.sumX[0, j] / Math.Sqrt(this.sumX[0, j] * this.sumX[0, j] + this.sumY[0, j] * this.sumY[0, j]);
                    double dCombinedMagnitude_dSumY = this.sumY[0, j] / Math.Sqrt(this.sumX[0, j] * this.sumX[0, j] + this.sumY[0, j] * this.sumY[0, j]);

                    double dCombinedAngle_dSumX = -deltay / (deltax * deltax + deltay * deltay);
                    double dCombinedAngle_dSumY = deltax / (deltax * deltax + deltay * deltay);

                    double dX1_dMagnitude = Math.Cos(angle);
                    double dY1_dMagnitude = Math.Sin(angle);

                    // Apply chain rule for dInput1
                    dInput1[0, j] += dOutput[0, j] * (
                        dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dX1 * dX1_dMagnitude +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dY1 * dY1_dMagnitude +
                        dCombinedMagnitude_dSumX * dSumX_dDeltaY * dDeltaY_dY1 * dY1_dMagnitude +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaX * dDeltaX_dX1 * dX1_dMagnitude);

                    dInput1[0, j] += dOutput[0, j + (this.input2.Rows / 2)] * (
                        dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dX1 * dX1_dMagnitude +
                        dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dY1 * dY1_dMagnitude +
                        dCombinedAngle_dSumX * dSumX_dDeltaY * dDeltaY_dY1 * dY1_dMagnitude +
                        dCombinedAngle_dSumY * dSumY_dDeltaX * dDeltaX_dX1 * dX1_dMagnitude);

                    double dX1_dAngle = -magnitude * Math.Sin(angle);
                    double dY1_dAngle = magnitude * Math.Cos(angle);

                    // Applying the chain rule for the angle component of input1
                    dInput1[0, j + this.input1.Cols / 2] += dOutput[0, j] * (
                        dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dX1 * dX1_dAngle +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dY1 * dY1_dAngle +
                        dCombinedMagnitude_dSumX * dSumX_dDeltaY * dDeltaY_dY1 * dY1_dAngle +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaX * dDeltaX_dX1 * dX1_dAngle);

                    dInput1[0, j + this.input1.Cols / 2] += dOutput[0, j + (this.input2.Rows / 2)] * (
                        dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dX1 * dX1_dAngle +
                        dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dY1 * dY1_dAngle +
                        dCombinedAngle_dSumX * dSumX_dDeltaY * dDeltaY_dY1 * dY1_dAngle +
                        dCombinedAngle_dSumY * dSumY_dDeltaX * dDeltaX_dX1 * dX1_dAngle);

                    double dX2_dWMagnitude = Math.Cos(wAngle);
                    double dY2_dWMagnitude = Math.Sin(wAngle);

                    // Apply chain rule for dInput2
                    dInput2[k, j] += dOutput[0, j] * (
                        dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dX2 * dX2_dWMagnitude +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dY2 * dY2_dWMagnitude +
                        dCombinedMagnitude_dSumX * dSumX_dDeltaY * dDeltaY_dY2 * dY2_dWMagnitude +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaX * dDeltaX_dX2 * dX2_dWMagnitude);

                    dInput2[k, j] += dOutput[0, j + (this.input2.Rows / 2)] * (
                        dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dX2 * dX2_dWMagnitude +
                        dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dY2 * dY2_dWMagnitude +
                        dCombinedAngle_dSumX * dSumX_dDeltaY * dDeltaY_dY2 * dY2_dWMagnitude +
                        dCombinedAngle_dSumY * dSumY_dDeltaX * dDeltaX_dX2 * dX2_dWMagnitude);


                    double dX2_dWAngle = -wMagnitude * Math.Sin(wAngle);
                    double dY2_dWAngle = wMagnitude * Math.Cos(wAngle);

                    dInput2[k, j + this.input2.Cols / 2] += dOutput[0, j] * (
                        dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dX2 * dX2_dWAngle +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dY2 * dY2_dWAngle +
                        dCombinedMagnitude_dSumX * dSumX_dDeltaY * dDeltaY_dY2 * dY2_dWAngle +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaX * dDeltaX_dX2 * dX2_dWAngle);

                    dInput2[k, j + this.input2.Cols / 2] += dOutput[0, j + (this.input2.Rows / 2)] * (
                        dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dX2 * dX2_dWAngle +
                        dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dY2 * dY2_dWAngle +
                        dCombinedAngle_dSumX * dSumX_dDeltaY * dDeltaY_dY2 * dY2_dWAngle +
                        dCombinedAngle_dSumY * dSumY_dDeltaX * dDeltaX_dX2 * dX2_dWAngle);

                    // Derivatives of delta components with respect to weight
                    double dDeltaX_dWeight = (this.weights[k, j] > 0) ? (x2 - x1) : (x1 - x2);
                    double dDeltaY_dWeight = (this.weights[k, j] > 0) ? (y2 - y1) : (y1 - y2);

                    // Apply chain rule for weights for magnitude and angle
                    dWeights[k, j] += dOutput[0, j] * (
                        dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dWeight +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dWeight +
                        dCombinedMagnitude_dSumX * dSumX_dDeltaY * dDeltaY_dWeight +
                        dCombinedMagnitude_dSumY * dSumY_dDeltaX * dDeltaX_dWeight);

                    dWeights[k, j] += dOutput[0, j + this.input1.Cols / 2] * (
                        dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dWeight +
                        dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dWeight +
                        dCombinedAngle_dSumX * dSumX_dDeltaY * dDeltaY_dWeight +
                        dCombinedAngle_dSumY * dSumY_dDeltaX * dDeltaX_dWeight);

                    double dResultMagnitude_dWeight = Math.Sqrt(deltax * deltax + deltay * deltay);
                    double resultAngle = Math.Atan2(deltay, deltax);
                    double dSumX_dResultMagnitude = Math.Cos(resultAngle);
                    double dSumY_dResultMagnitude = Math.Sin(resultAngle);

                    // Apply chain rule for weights for magnitude and angle
                    dWeights[k, j] += dOutput[0, j] * (
                        dCombinedMagnitude_dSumX * dSumX_dResultMagnitude * dResultMagnitude_dWeight +
                        dCombinedMagnitude_dSumY * dSumY_dResultMagnitude * dResultMagnitude_dWeight);

                    dWeights[k, j] += dOutput[0, j + this.input1.Cols / 2] * (
                        dCombinedAngle_dSumX * dSumX_dResultMagnitude * dResultMagnitude_dWeight +
                        dCombinedAngle_dSumY * dSumY_dResultMagnitude * dResultMagnitude_dWeight);

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
