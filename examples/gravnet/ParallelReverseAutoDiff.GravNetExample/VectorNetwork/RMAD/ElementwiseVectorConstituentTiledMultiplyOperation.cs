//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorConstituentTiledMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;
    using ParallelReverseAutoDiff.GravNetExample.Common;

    /// <summary>
    /// Element-wise tiled multiplication operation.
    /// </summary>
    public class ElementwiseVectorConstituentTiledMultiplyOperation : Operation
    {
        private Matrix[,] input1;
        private Matrix[,] input2;
        private Matrix[,] weights;
        private Matrix[,] sumX;
        private Matrix[,] sumY;
        private Matrix[,] output;
        private Matrix[,] dInput1;
        private Matrix[,] dInput2;
        private Matrix[,] dWeights;
        private CalculatedValues[,][,] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorConstituentTiledMultiplyOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector constituent multiply function.
        /// </summary>
        /// <param name="input1">The first input to the vector constituent multiply operation.</param>
        /// <param name="input2">The second input to the vector constituent multiply operation.</param>
        /// <returns>The output of the vector constituent multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            var brokenInput1 = CommonMatrixUtils.BreakIntoSections(input1, 8);
            var brokenInput2 = CommonMatrixUtils.BreakIntoSections(input2, 8);
            var brokenWeights = CommonMatrixUtils.BreakIntoSections(weights, 8);
            this.input1 = brokenInput1;
            this.input2 = brokenInput2;
            this.weights = brokenWeights;

            Parallel.For(0, brokenInput1.GetLength(0), i =>
            {
                for (int j = 0; j < brokenInput2.GetLength(1); j++)
                {
                    this.InnerForward(i, j, brokenInput1[i, j], brokenInput2[i, j], brokenWeights[i, j]);
                }
            });

            this.Output = CommonMatrixUtils.PieceTogether(this.output);
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            this.dInput1 = new Matrix[this.input1.GetLength(0), this.input1.GetLength(1)];
            this.dInput2 = new Matrix[this.input2.GetLength(0), this.input2.GetLength(1)];
            this.dWeights = new Matrix[this.weights.GetLength(0), this.weights.GetLength(1)];
            var dOutputSections = CommonMatrixUtils.BreakIntoSections(dOutput, 8);

            Parallel.For(0, this.dInput1.GetLength(0), i =>
            {
                for (int j = 0; j < this.dInput2.GetLength(1); j++)
                {
                    this.InnerBackward(i, j, this.dInput1[i, j], this.dInput2[i, j], this.dWeights[i, j], dOutputSections[i, j]);
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(CommonMatrixUtils.PieceTogether(this.dInput1))
                .AddInputGradient(CommonMatrixUtils.PieceTogether(this.dInput2))
                .AddInputGradient(CommonMatrixUtils.PieceTogether(this.dWeights))
                .Build();
        }

        private void InnerForward(int ii, int jj, Matrix input1, Matrix input2, Matrix weights)
        {
            this.calculatedValues[ii, jj] = new CalculatedValues[input1.Rows, input2.Cols / 2];

            this.output[ii, jj] = new Matrix(input1.Rows, input2.Cols);
            this.sumX[ii, jj] = new Matrix(input1.Rows, input2.Cols / 2);
            this.sumY[ii, jj] = new Matrix(input1.Rows, input2.Cols / 2);

            for (int i = 0; i < input1.Rows; ++i) 
            {
                for (int j = 0; j < input2.Cols / 2; j++)
                {
                    double sumX = 0.0d;
                    double sumY = 0.0d;

                    double[] dDeltaX_dX1 = new double[input2.Rows / 2];
                    double[] dDeltaY_dY1 = new double[input2.Rows / 2];
                    double[] dDeltaX_dX2 = new double[input2.Rows / 2];
                    double[] dDeltaY_dY2 = new double[input2.Rows / 2];
                    double[] dSumX_dDeltaX = new double[input2.Rows / 2];
                    double[] dSumX_dDeltaY = new double[input2.Rows / 2];
                    double[] dSumY_dDeltaX = new double[input2.Rows / 2];
                    double[] dSumY_dDeltaY = new double[input2.Rows / 2];
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
                    double[] dSumX_dResultMagnitude = new double[input2.Rows / 2];
                    double[] dSumY_dResultMagnitude = new double[input2.Rows / 2];
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

                        dSumX_dDeltaX[k] = dLocalSumX_dDeltaX;
                        dSumX_dDeltaY[k] = dLocalSumX_dDeltaY;
                        dSumY_dDeltaX[k] = dLocalSumY_dDeltaX;
                        dSumY_dDeltaY[k] = dLocalSumY_dDeltaY;

                        // Derivatives of delta components with respect to inputs
                        dDeltaX_dX1[k] = weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        dDeltaY_dY1[k] = weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        dDeltaX_dX2[k] = weights[k, j] > 0 ? 1 : -1; // Depending on weight sign
                        dDeltaY_dY2[k] = weights[k, j] > 0 ? 1 : -1; // Depending on weight sign

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
                        dSumX_dResultMagnitude[k] = Math.Cos(resultAngle);
                        dSumY_dResultMagnitude[k] = Math.Sin(resultAngle);
                    }

                    this.sumX[ii, jj][i, j] = sumX;
                    this.sumY[ii, jj][i, j] = sumY;

                    // Analytically determined gradients for combined magnitude
                    double magSumXY = (this.sumX[ii, jj][i, j] * this.sumX[ii, jj][i, j]) + (this.sumY[ii, jj][i, j] * this.sumY[ii, jj][i, j]);
                    double dCombinedMagnitude_dSumX = this.sumX[ii, jj][i, j] / Math.Sqrt(magSumXY);
                    double dCombinedMagnitude_dSumY = this.sumY[ii, jj][i, j] / Math.Sqrt(magSumXY);

                    double dCombinedAngle_dSumX = -this.sumY[ii, jj][i, j] / magSumXY;
                    double dCombinedAngle_dSumY = this.sumX[ii, jj][i, j] / magSumXY;

                    for (int k = 0; k < input2.Rows / 2; k++)
                    {
                        dInputMag_dOutputMag +=
                            dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k];

                        dInput2Mag_dOutputMag +=
                            dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX2_dWMagnitude[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY2_dWMagnitude[k] +
                            dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY2_dWMagnitude[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX2_dWMagnitude[k];

                        dInputMag_dOutputAngle +=
                            dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k];

                        dInput2Mag_dOutputAngle +=
                            dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWMagnitude[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWMagnitude[k] +
                            dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWMagnitude[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWMagnitude[k];

                        dInputAngle_dOutputMag +=
                            dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k];

                        dInput2Angle_dOutputMag +=
                            dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k];

                        dInputAngle_dOutputAngle +=
                            dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k];

                        dInput2Angle_dOutputAngle +=
                            dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k];

                        dWeight_dOutputMag +=
                            dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dWeight[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dWeight[k] +
                            dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dWeight[k] +
                            dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dWeight[k] +
                            dCombinedMagnitude_dSumX * dSumX_dResultMagnitude[k] * dResultMagnitude_dWeight[k] +
                            dCombinedMagnitude_dSumY * dSumY_dResultMagnitude[k] * dResultMagnitude_dWeight[k];

                        dWeight_dOutputAngle +=
                            dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dWeight[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dWeight[k] +
                            dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dWeight[k] +
                            dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dWeight[k] +
                            dCombinedAngle_dSumX * dSumX_dResultMagnitude[k] * dResultMagnitude_dWeight[k] +
                            dCombinedAngle_dSumY * dSumY_dResultMagnitude[k] * dResultMagnitude_dWeight[k];
                    }

                    this.calculatedValues[ii, jj][i, j].DInputMag_dOutputMag = dInputMag_dOutputMag;
                    this.calculatedValues[ii, jj][i, j].DInputMag_dOutputAngle = dInputMag_dOutputAngle;
                    this.calculatedValues[ii, jj][i, j].DInputAngle_dOutputMag = dInputAngle_dOutputMag;
                    this.calculatedValues[ii, jj][i, j].DInputAngle_dOutputAngle = dInputAngle_dOutputAngle;
                    this.calculatedValues[ii, jj][i, j].DInput2Mag_dOutputMag = dInput2Mag_dOutputMag;
                    this.calculatedValues[ii, jj][i, j].DInput2Mag_dOutputAngle = dInput2Mag_dOutputAngle;
                    this.calculatedValues[ii, jj][i, j].DInput2Angle_dOutputMag = dInput2Angle_dOutputMag;
                    this.calculatedValues[ii, jj][i, j].DInput2Angle_dOutputAngle = dInput2Angle_dOutputAngle;
                    this.calculatedValues[ii, jj][i, j].DWeight_dOutputMag = dWeight_dOutputMag;
                    this.calculatedValues[ii, jj][i, j].DWeight_dOutputAngle = dWeight_dOutputAngle;

                    this.output[ii, jj][i, j] = Math.Sqrt((sumX * sumX) + (sumY * sumY)); // Magnitude
                    this.output[ii, jj][i, j + (input2.Cols / 2)] = Math.Atan2(sumY, sumX); // Angle in radians
                }
            }
        }

        private void InnerBackward(int ii, int jj, Matrix input1, Matrix input2, Matrix weights, Matrix dOutput)
        {
            // Initialize gradient matrices
            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            // Loop through each element in input1
            for (int i = 0; i < input1.Rows; ++i)
            {
                // Loop through each half of the columns in input1 (representing magnitudes and angles)
                for (int j = 0; j < input2.Cols / 2; j++)
                {
                    var calculatedValues = this.calculatedValues[ii, jj][i, j];
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
            }
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
