//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorConstituentMonoMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
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
    public class ElementwiseVectorConstituentMonoMultiplyOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private Matrix sumX;
        private Matrix sumY;
        private CalculatedValues[,] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorConstituentMonoMultiplyOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector constituent multiply function.
        /// </summary>
        /// <param name="input1">The first input to the vector constituent multiply operation.</param>
        /// <param name="input2">The second input to the vector constituent multiply operation.</param>
        /// <returns>The output of the vector constituent multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;
            this.calculatedValues = new CalculatedValues[input1.Rows, input2.Cols / 2];

            this.Output = new Matrix(input1.Rows,input2.Cols);
            this.sumX = new Matrix(input1.Rows, input2.Cols / 2);
            this.sumY = new Matrix(input1.Rows, input2.Cols / 2);

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input2.Cols / 2; j++)
                {
                    double sumX = 0.0d;
                    double sumY = 0.0d;

                    double dSumX_dDeltaX = 0.0d;
                    double dSumX_dDeltaY = 0.0d;
                    double dSumY_dDeltaX = 0.0d;
                    double dSumY_dDeltaY = 0.0d;
                    double dSumX_dResultMagnitude = 0.0d;
                    double dSumY_dResultMagnitude = 0.0d;

                    double[] dDeltaX_dX1 = new double[input2.Rows / 2];
                    double[] dDeltaY_dY1 = new double[input2.Rows / 2];
                    double[] dDeltaX_dX2 = new double[input2.Rows / 2];
                    double[] dDeltaY_dY2 = new double[input2.Rows / 2];
                    double[] dDeltaX_dWeight = new double[input2.Rows / 2];
                    double[] dDeltaY_dWeight = new double[input2.Rows / 2];
                    double[] dX1_dMagnitude = new double[input2.Rows / 2];
                    double[] dY1_dMagnitude = new double[input2.Rows / 2];
                    double[] dX1_dAngle = new double[input2.Rows / 2];
                    double[] dY1_dAngle = new double[input2.Rows / 2];
                    double[] dX2_dWMagnitude = new double[input2.Rows / 2];
                    double[] dY2_dWMagnitude = new double[input2.Rows / 2];
                    double[] dX2_dWAngle = new double[input2.Rows / 2];
                    double[] dY2_dWAngle = new double[input2.Rows / 2];
                    double[] dResultMagnitude_dWeight = new double[input2.Rows / 2];

                    double dInputMag_dOutputMag = 0.0d;
                    double dInputMag_dOutputAngle = 0.0d;
                    double dInputAngle_dOutputMag = 0.0d;
                    double dInputAngle_dOutputAngle = 0.0d;
                    double dInput2Mag_dOutputMag = 0.0d;
                    double dInput2Mag_dOutputAngle = 0.0d;
                    double dInput2Angle_dOutputMag = 0.0d;
                    double dInput2Angle_dOutputAngle = 0.0d;
                    double dWeight_dOutputMag = 0.0d;
                    double dWeight_dOutputAngle = 0.0d;

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

                        double deltaXYSquared = (deltax * deltax) + (deltay * deltay);

                        // Compute resultant vector magnitude and angle
                        double resultMagnitude = Math.Sqrt(deltaXYSquared) * weights[k, j];
                        double resultAngle = Math.Atan2(deltay, deltax);

                        double dResultMagnitude_dDeltaX = (deltax * weights[k, j]) / Math.Sqrt(deltaXYSquared);
                        double dResultMagnitude_dDeltaY = (deltay * weights[k, j]) / Math.Sqrt(deltaXYSquared);
                        double dResultAngle_dDeltaX = -deltay / deltaXYSquared;
                        double dResultAngle_dDeltaY = deltax / deltaXYSquared;

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

                        // Derivatives of delta components with respect to inputs
                        dDeltaX_dX1[k] = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        dDeltaY_dY1[k] = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        dDeltaX_dX2[k] = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign
                        dDeltaY_dY2[k] = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign

                        dX1_dMagnitude[k] = Math.Cos(angle);
                        dY1_dMagnitude[k] = Math.Sin(angle);

                        dX1_dAngle[k] = -magnitude * Math.Sin(angle);
                        dY1_dAngle[k] = magnitude * Math.Cos(angle);

                        dX2_dWMagnitude[k] = Math.Cos(wAngle);
                        dY2_dWMagnitude[k] = Math.Sin(wAngle);

                        dX2_dWAngle[k] = -wMagnitude * Math.Sin(wAngle);
                        dY2_dWAngle[k] = wMagnitude * Math.Cos(wAngle);

                        // Derivatives of delta components with respect to weight
                        dDeltaX_dWeight[k] = (weights[k, j] > 0) ? (x2 - x1) : (x1 - x2);
                        dDeltaY_dWeight[k] = (weights[k, j] > 0) ? (y2 - y1) : (y1 - y2);

                        dResultMagnitude_dWeight[k] = Math.Sqrt(deltaXYSquared);
                        dSumX_dResultMagnitude += Math.Cos(resultAngle);
                        dSumY_dResultMagnitude += Math.Sin(resultAngle);
                    }

                    this.sumX[i, j] = sumX;
                    this.sumY[i, j] = sumY;

                    // Analytically determined gradients for combined magnitude
                    double magSumXY = (this.sumX[i, j] * this.sumX[i, j]) + (this.sumY[i, j] * this.sumY[i, j]);
                    double dCombinedMagnitude_dSumX = this.sumX[i, j] / Math.Sqrt(magSumXY);
                    double dCombinedMagnitude_dSumY = this.sumY[i, j] / Math.Sqrt(magSumXY);

                    double dCombinedAngle_dSumX = -this.sumY[i, j] / magSumXY;
                    double dCombinedAngle_dSumY = this.sumX[i, j] / magSumXY;

                    double a1 = dCombinedMagnitude_dSumX * dSumX_dDeltaX;
                    double a2 = dCombinedMagnitude_dSumY * dSumY_dDeltaY;
                    double a3 = dCombinedMagnitude_dSumX * dSumX_dDeltaY;
                    double a4 = dCombinedMagnitude_dSumY * dSumY_dDeltaX;

                    double b1 = dCombinedAngle_dSumX * dSumX_dDeltaX;
                    double b2 = dCombinedAngle_dSumY * dSumY_dDeltaY;
                    double b3 = dCombinedAngle_dSumX * dSumX_dDeltaY;
                    double b4 = dCombinedAngle_dSumY * dSumY_dDeltaX;

                    double c1 = dCombinedMagnitude_dSumX * dSumX_dDeltaX;
                    double c2 = dCombinedMagnitude_dSumY * dSumY_dDeltaY;
                    double c3 = dCombinedMagnitude_dSumX * dSumX_dDeltaY;
                    double c4 = dCombinedMagnitude_dSumY * dSumY_dDeltaX;

                    double d1 = dCombinedAngle_dSumX * dSumX_dDeltaX;
                    double d2 = dCombinedAngle_dSumY * dSumY_dDeltaY;
                    double d3 = dCombinedAngle_dSumX * dSumX_dDeltaY;
                    double d4 = dCombinedAngle_dSumY * dSumY_dDeltaX;

                    for (int k = 0; k < input2.Rows / 2; k++)
                    {
                        dInputMag_dOutputMag +=
                            a1 * dDeltaX_dX1[k] * dX1_dMagnitude[k] +
                            a2 * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            a3 * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            a4 * dDeltaX_dX1[k] * dX1_dMagnitude[k];

                        dInput2Mag_dOutputMag += 
                            a1 * dDeltaX_dX1[k] * dX2_dWMagnitude[k] +
                            a2 * dDeltaY_dY1[k] * dY2_dWMagnitude[k] +
                            a3 * dDeltaY_dY1[k] * dY2_dWMagnitude[k] +
                            a4 * dDeltaX_dX1[k] * dX2_dWMagnitude[k];

                        dInputMag_dOutputAngle +=
                            b1 * dDeltaX_dX1[k] * dX1_dMagnitude[k] +
                            b2 * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            b3 * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            b4 * dDeltaX_dX1[k] * dX1_dMagnitude[k];

                        dInput2Mag_dOutputAngle +=
                            b1 * dDeltaX_dX2[k] * dX2_dWMagnitude[k] +
                            b2 * dDeltaY_dY2[k] * dY2_dWMagnitude[k] +
                            b3 * dDeltaY_dY2[k] * dY2_dWMagnitude[k] +
                            b4 * dDeltaX_dX2[k] * dX2_dWMagnitude[k];

                        dInputAngle_dOutputMag +=
                            c1 * dDeltaX_dX1[k] * dX1_dAngle[k] +
                            c2 * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            c3 * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            c4 * dDeltaX_dX1[k] * dX1_dAngle[k];

                        dInput2Angle_dOutputMag +=
                            c1 * dDeltaX_dX2[k] * dX2_dWAngle[k] +
                            c2 * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            c3 * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            c4 * dDeltaX_dX2[k] * dX2_dWAngle[k];

                        dInputAngle_dOutputAngle +=
                            d1 * dDeltaX_dX1[k] * dX1_dAngle[k] +
                            d2 * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            d3 * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            d4 * dDeltaX_dX1[k] * dX1_dAngle[k];

                        dInput2Angle_dOutputAngle +=
                            d1 * dDeltaX_dX2[k] * dX2_dWAngle[k] +
                            d2 * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            d3 * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            d4 * dDeltaX_dX2[k] * dX2_dWAngle[k];

                        dWeight_dOutputMag +=
                            dCombinedMagnitude_dSumX * dSumX_dDeltaX * dDeltaX_dWeight[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaY * dDeltaY_dWeight[k] +
                            dCombinedMagnitude_dSumX * dSumX_dDeltaY * dDeltaY_dWeight[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaX * dDeltaX_dWeight[k] +
                            dCombinedMagnitude_dSumX * dSumX_dResultMagnitude * dResultMagnitude_dWeight[k] +
                            dCombinedMagnitude_dSumY * dSumY_dResultMagnitude * dResultMagnitude_dWeight[k];

                        dWeight_dOutputAngle +=
                            dCombinedAngle_dSumX * dSumX_dDeltaX * dDeltaX_dWeight[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaY * dDeltaY_dWeight[k] +
                            dCombinedAngle_dSumX * dSumX_dDeltaY * dDeltaY_dWeight[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaX * dDeltaX_dWeight[k] +
                            dCombinedAngle_dSumX * dSumX_dResultMagnitude * dResultMagnitude_dWeight[k] +
                            dCombinedAngle_dSumY * dSumY_dResultMagnitude * dResultMagnitude_dWeight[k];
                    }

                    this.calculatedValues[i, j].DInputMag_dOutputMag = dInputMag_dOutputMag;
                    this.calculatedValues[i, j].DInputMag_dOutputAngle = dInputMag_dOutputAngle;
                    this.calculatedValues[i, j].DInputAngle_dOutputMag = dInputAngle_dOutputMag;
                    this.calculatedValues[i, j].DInputAngle_dOutputAngle = dInputAngle_dOutputAngle;
                    this.calculatedValues[i, j].DInput2Mag_dOutputMag = dInput2Mag_dOutputMag;
                    this.calculatedValues[i, j].DInput2Mag_dOutputAngle = dInput2Mag_dOutputAngle;
                    this.calculatedValues[i, j].DInput2Angle_dOutputMag = dInput2Angle_dOutputMag;
                    this.calculatedValues[i, j].DInput2Angle_dOutputAngle = dInput2Angle_dOutputAngle;
                    this.calculatedValues[i, j].DWeight_dOutputMag = dWeight_dOutputMag;
                    this.calculatedValues[i, j].DWeight_dOutputAngle = dWeight_dOutputAngle;

                    this.Output[i, j] = Math.Sqrt((sumX * sumX) + (sumY * sumY)); // Magnitude
                    this.Output[i, j + (input2.Cols / 2)] = Math.Atan2(sumY, sumX); // Angle in radians
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
                    var calculatedValues = this.calculatedValues[i, j];
                    dInput1[i, j] += dOutput[i, j] * calculatedValues.DInputMag_dOutputMag;
                    dInput1[i, j] += dOutput[i, j + (input2.Cols / 2)] * calculatedValues.DInputMag_dOutputAngle;
                    dInput1[i, j + (input1.Cols / 2)] += dOutput[i, j] * calculatedValues.DInputAngle_dOutputMag;
                    dInput1[i, j + (input1.Cols / 2)] += dOutput[i, j + (input2.Cols / 2)] * calculatedValues.DInputAngle_dOutputAngle;

                    dInput2[i, j] += dOutput[i, j] * calculatedValues.DInput2Mag_dOutputMag;
                    dInput2[i, j] += dOutput[i, j + (input2.Cols / 2)] * calculatedValues.DInput2Mag_dOutputAngle;
                    dInput2[i, j + (input2.Cols / 2)] += dOutput[i, j] * calculatedValues.DInput2Angle_dOutputMag;
                    dInput2[i, j + (input2.Cols / 2)] += dOutput[i, j + (input2.Cols / 2)] * calculatedValues.DInput2Angle_dOutputAngle;

                    dWeights[i, j] += dOutput[i, j] * calculatedValues.DWeight_dOutputMag;
                    dWeights[i, j] += dOutput[i, j + (input2.Cols / 2)] * calculatedValues.DWeight_dOutputAngle;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

        private struct CalculatedValues
        {
            public double DInputMag_dOutputMag { get; internal set; }

            public double DInputMag_dOutputAngle { get; internal set; }

            public double DInputAngle_dOutputMag { get; internal set; }

            public double DInputAngle_dOutputAngle { get; internal set; }

            public double DInput2Mag_dOutputMag { get; internal set; }

            public double DInput2Mag_dOutputAngle { get; internal set; }

            public double DInput2Angle_dOutputMag { get; internal set; }

            public double DInput2Angle_dOutputAngle { get; internal set; }

            public double DWeight_dOutputMag { get; internal set; }

            public double DWeight_dOutputAngle { get; internal set; }
        }
    }
}
