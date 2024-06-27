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
    /// Element-wise vector constituent multiplication operation.
    /// </summary>
    public class ElementwiseVectorConstituentMultiplyOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private Matrix sumX;
        private Matrix sumY;
        private CalculatedValues calculatedValues;

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
        /// Performs the forward operation for the vector constituent multiply function.
        /// </summary>
        /// <param name="input1">The first input to the vector constituent multiply operation.</param>
        /// <param name="input2">The second input to the vector constituent multiply operation.</param>
        /// <param name="weights">The weights to the vector constituent multiply operation.</param>
        /// <returns>The output of the vector constituent multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(input1.Rows, input2.Cols);
            this.sumX = new Matrix(input1.Rows, input2.Cols / 2);
            this.sumY = new Matrix(input1.Rows, input2.Cols / 2);

            float[,] dInputMag_dOutputMag = new float[input1.Rows, input2.Rows];
            float[,] dInputMag_dOutputAngle = new float[input1.Rows, input2.Rows];
            float[,] dInputAngle_dOutputMag = new float[input1.Rows, input2.Rows];
            float[,] dInputAngle_dOutputAngle = new float[input1.Rows, input2.Rows];
            float[,] dInput2Mag_dOutputMag = new float[input2.Rows, input2.Cols / 2];
            float[,] dInput2Mag_dOutputAngle = new float[input2.Rows, input2.Cols / 2];
            float[,] dInput2Angle_dOutputMag = new float[input2.Rows, input2.Cols / 2];
            float[,] dInput2Angle_dOutputAngle = new float[input2.Rows, input2.Cols / 2];
            float[,] dWeight_dOutputMag = new float[input2.Rows, input2.Cols / 2];
            float[,] dWeight_dOutputAngle = new float[input2.Rows, input2.Cols / 2];

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input2.Cols / 2; j++)
                {
                    float sumX = 0.0f;
                    float sumY = 0.0f;

                    float[] dDeltaX_dX1 = new float[input2.Rows];
                    float[] dDeltaY_dY1 = new float[input2.Rows];
                    float[] dDeltaX_dX2 = new float[input2.Rows];
                    float[] dDeltaY_dY2 = new float[input2.Rows];
                    float[] dSumX_dDeltaX = new float[input2.Rows];
                    float[] dSumX_dDeltaY = new float[input2.Rows];
                    float[] dSumY_dDeltaX = new float[input2.Rows];
                    float[] dSumY_dDeltaY = new float[input2.Rows];
                    float[] dDeltaX_dWeight = new float[input2.Rows];
                    float[] dDeltaY_dWeight = new float[input2.Rows];
                    float[] dX1_dMagnitude = new float[input2.Rows];
                    float[] dY1_dMagnitude = new float[input2.Rows];
                    float[] dX1_dAngle = new float[input2.Rows];
                    float[] dY1_dAngle = new float[input2.Rows];
                    float[] dX2_dWMagnitude = new float[input2.Rows];
                    float[] dY2_dWMagnitude = new float[input2.Rows];
                    float[] dX2_dWAngle = new float[input2.Rows];
                    float[] dY2_dWAngle = new float[input2.Rows];
                    float[] dSumX_dResultMagnitude = new float[input2.Rows];
                    float[] dSumY_dResultMagnitude = new float[input2.Rows];
                    float[] dResultMagnitude_dWeight = new float[input2.Rows];

                    for (int k = 0; k < input2.Rows; k++)
                    {
                        // Accessing the magnitudes and angles from the concatenated matrices
                        float magnitude = input1[i, k];
                        float angle = input1[i, k + (input1.Cols / 2)];

                        float wMagnitude = input2[k, j];
                        float wAngle = input2[k, j + (input2.Cols / 2)];

                        // Compute vector components
                        float x1 = magnitude * PradMath.Cos(angle);
                        float y1 = magnitude * PradMath.Sin(angle);
                        float x2 = wMagnitude * PradMath.Cos(wAngle);
                        float y2 = wMagnitude * PradMath.Sin(wAngle);

                        // Select vector direction based on weight
                        float deltax = weights[k, j] > 0 ? x2 - x1 : x1 - x2;
                        float deltay = weights[k, j] > 0 ? y2 - y1 : y1 - y2;

                        float deltaXYSquared = (deltax * deltax) + (deltay * deltay);

                        // Compute resultant vector magnitude and angle
                        float resultMagnitude = PradMath.Sqrt(deltaXYSquared) * weights[k, j];
                        float resultAngle = PradMath.Atan2(deltay, deltax);

                        float dResultMagnitude_dDeltaX = (deltax * weights[k, j]) / PradMath.Sqrt(deltaXYSquared);
                        float dResultMagnitude_dDeltaY = (deltay * weights[k, j]) / PradMath.Sqrt(deltaXYSquared);
                        float dResultAngle_dDeltaX = -deltay / deltaXYSquared;
                        float dResultAngle_dDeltaY = deltax / deltaXYSquared;

                        float localSumX = resultMagnitude * PradMath.Cos(resultAngle);
                        float localSumY = resultMagnitude * PradMath.Sin(resultAngle);

                        float dLocalSumX_dResultMagnitude = PradMath.Cos(resultAngle);
                        float dLocalSumY_dResultMagnitude = PradMath.Sin(resultAngle);

                        float dLocalSumX_dResultAngle = -resultMagnitude * PradMath.Sin(resultAngle);
                        float dLocalSumY_dResultAngle = resultMagnitude * PradMath.Cos(resultAngle);

                        float dLocalSumX_dDeltaX = (dLocalSumX_dResultMagnitude * dResultMagnitude_dDeltaX)
                            + (dLocalSumX_dResultAngle * dResultAngle_dDeltaX);
                        float dLocalSumX_dDeltaY = (dLocalSumX_dResultMagnitude * dResultMagnitude_dDeltaY)
                            + (dLocalSumX_dResultAngle * dResultAngle_dDeltaY);
                        float dLocalSumY_dDeltaX = (dLocalSumY_dResultMagnitude * dResultMagnitude_dDeltaX)
                            + (dLocalSumY_dResultAngle * dResultAngle_dDeltaX);
                        float dLocalSumY_dDeltaY = (dLocalSumY_dResultMagnitude * dResultMagnitude_dDeltaY)
                            + (dLocalSumY_dResultAngle * dResultAngle_dDeltaY);

                        sumX += localSumX;
                        sumY += localSumY;

                        dSumX_dDeltaX[k] = dLocalSumX_dDeltaX;
                        dSumX_dDeltaY[k] = dLocalSumX_dDeltaY;
                        dSumY_dDeltaX[k] = dLocalSumY_dDeltaX;
                        dSumY_dDeltaY[k] = dLocalSumY_dDeltaY;

                        // Derivatives of delta components with respect to inputs
                        dDeltaX_dX1[k] = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        dDeltaY_dY1[k] = this.weights[k, j] > 0 ? -1 : 1; // Depending on weight sign
                        dDeltaX_dX2[k] = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign
                        dDeltaY_dY2[k] = this.weights[k, j] > 0 ? 1 : -1; // Depending on weight sign

                        dX1_dMagnitude[k] = PradMath.Cos(angle);
                        dY1_dMagnitude[k] = PradMath.Sin(angle);

                        dX1_dAngle[k] = -magnitude * PradMath.Sin(angle);
                        dY1_dAngle[k] = magnitude * PradMath.Cos(angle);

                        dX2_dWMagnitude[k] = PradMath.Cos(wAngle);
                        dY2_dWMagnitude[k] = PradMath.Sin(wAngle);

                        dX2_dWAngle[k] = -wMagnitude * PradMath.Sin(wAngle);
                        dY2_dWAngle[k] = wMagnitude * PradMath.Cos(wAngle);

                        // Derivatives of delta components with respect to weight
                        dDeltaX_dWeight[k] = (weights[k, j] > 0) ? (x2 - x1) : (x1 - x2);
                        dDeltaY_dWeight[k] = (weights[k, j] > 0) ? (y2 - y1) : (y1 - y2);

                        dResultMagnitude_dWeight[k] = PradMath.Sqrt(deltaXYSquared);
                        dSumX_dResultMagnitude[k] = PradMath.Cos(resultAngle);
                        dSumY_dResultMagnitude[k] = PradMath.Sin(resultAngle);
                    }

                    this.sumX[i, j] = sumX;
                    this.sumY[i, j] = sumY;

                    // Analytically determined gradients for combined magnitude
                    float magSumXY = (this.sumX[i, j] * this.sumX[i, j]) + (this.sumY[i, j] * this.sumY[i, j]);
                    float dCombinedMagnitude_dSumX = this.sumX[i, j] / PradMath.Sqrt(magSumXY);
                    float dCombinedMagnitude_dSumY = this.sumY[i, j] / PradMath.Sqrt(magSumXY);

                    float dCombinedAngle_dSumX = -this.sumY[i, j] / magSumXY;
                    float dCombinedAngle_dSumY = this.sumX[i, j] / magSumXY;

                    for (int k = 0; k < input2.Rows; k++)
                    {
                        dInputMag_dOutputMag[i, k] +=
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k]) +
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k]);

                        dInput2Mag_dOutputMag[k, j] +=
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX2_dWMagnitude[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY2_dWMagnitude[k]) +
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY2_dWMagnitude[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX2_dWMagnitude[k]);

                        dInputMag_dOutputAngle[i, k] +=
                            (dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k]) +
                            (dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dMagnitude[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dMagnitude[k]);

                        dInput2Mag_dOutputAngle[k, j] +=
                            (dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWMagnitude[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWMagnitude[k]) +
                            (dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWMagnitude[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWMagnitude[k]);

                        dInputAngle_dOutputMag[i, k] +=
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k]) +
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k]);

                        dInput2Angle_dOutputMag[k, j] +=
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k]) +
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k]);

                        dInputAngle_dOutputAngle[i, k] +=
                            (dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k]) +
                            (dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY1[k] * dY1_dAngle[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX1[k] * dX1_dAngle[k]);

                        dInput2Angle_dOutputAngle[k, j] +=
                            (dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k]) +
                            (dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dY2[k] * dY2_dWAngle[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dX2[k] * dX2_dWAngle[k]);

                        dWeight_dOutputMag[k, j] +=
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaX[k] * dDeltaX_dWeight[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaY[k] * dDeltaY_dWeight[k]) +
                            (dCombinedMagnitude_dSumX * dSumX_dDeltaY[k] * dDeltaY_dWeight[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dDeltaX[k] * dDeltaX_dWeight[k]) +
                            (dCombinedMagnitude_dSumX * dSumX_dResultMagnitude[k] * dResultMagnitude_dWeight[k]) +
                            (dCombinedMagnitude_dSumY * dSumY_dResultMagnitude[k] * dResultMagnitude_dWeight[k]);

                        dWeight_dOutputAngle[k, j] +=
                            (dCombinedAngle_dSumX * dSumX_dDeltaX[k] * dDeltaX_dWeight[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaY[k] * dDeltaY_dWeight[k]) +
                            (dCombinedAngle_dSumX * dSumX_dDeltaY[k] * dDeltaY_dWeight[k]) +
                            (dCombinedAngle_dSumY * dSumY_dDeltaX[k] * dDeltaX_dWeight[k]) +
                            (dCombinedAngle_dSumX * dSumX_dResultMagnitude[k] * dResultMagnitude_dWeight[k]) +
                            (dCombinedAngle_dSumY * dSumY_dResultMagnitude[k] * dResultMagnitude_dWeight[k]);
                    }

                    this.Output[i, j] = PradMath.Sqrt((sumX * sumX) + (sumY * sumY)); // Magnitude
                    this.Output[i, j + (input2.Cols / 2)] = PradMath.Atan2(sumY, sumX); // Angle in radians
                }
            });

            this.calculatedValues = new CalculatedValues
            {
                DInputMag_dOutputMag = dInputMag_dOutputMag,
                DInputMag_dOutputAngle = dInputMag_dOutputAngle,
                DInputAngle_dOutputMag = dInputAngle_dOutputMag,
                DInputAngle_dOutputAngle = dInputAngle_dOutputAngle,
                DInput2Mag_dOutputMag = dInput2Mag_dOutputMag,
                DInput2Mag_dOutputAngle = dInput2Mag_dOutputAngle,
                DInput2Angle_dOutputMag = dInput2Angle_dOutputMag,
                DInput2Angle_dOutputAngle = dInput2Angle_dOutputAngle,
                DWeight_dOutputMag = dWeight_dOutputMag,
                DWeight_dOutputAngle = dWeight_dOutputAngle,
            };

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Initialize gradient matrices
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            // Loop through each element in input1
            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int k = 0; k < this.input2.Rows; k++)
                {
                    for (int j = 0; j < this.input2.Cols / 2; j++)
                    {
                        dInput1[i, k] += dOutput[i, j] * this.calculatedValues.DInputMag_dOutputMag[i, k];
                        dInput1[i, k] += dOutput[i, j + (this.input2.Cols / 2)] * this.calculatedValues.DInputMag_dOutputAngle[i, k];
                        dInput1[i, k + (this.input1.Cols / 2)] += dOutput[i, j] * this.calculatedValues.DInputAngle_dOutputMag[i, k];
                        dInput1[i, k + (this.input1.Cols / 2)] += dOutput[i, j + (this.input2.Cols / 2)] * this.calculatedValues.DInputAngle_dOutputAngle[i, k];
                    }
                }
            });

            Parallel.For(0, this.input2.Rows, k =>
            {
                for (int j = 0; j < this.input2.Cols / 2; j++)
                {
                    for (int i = 0; i < this.input1.Rows; ++i)
                    {
                        dInput2[k, j] += dOutput[i, j] * this.calculatedValues.DInput2Mag_dOutputMag[k, j];
                        dInput2[k, j] += dOutput[i, j + (this.input2.Cols / 2)] * this.calculatedValues.DInput2Mag_dOutputAngle[k, j];
                        dInput2[k, j + (this.input2.Cols / 2)] += dOutput[i, j] * this.calculatedValues.DInput2Angle_dOutputMag[k, j];
                        dInput2[k, j + (this.input2.Cols / 2)] += dOutput[i, j + (this.input2.Cols / 2)] * this.calculatedValues.DInput2Angle_dOutputAngle[k, j];

                        dWeights[k, j] += dOutput[i, j] * this.calculatedValues.DWeight_dOutputMag[k, j];
                        dWeights[k, j] += dOutput[i, j + (this.input2.Cols / 2)] * this.calculatedValues.DWeight_dOutputAngle[k, j];
                    }
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
            public float[,] DInputMag_dOutputMag { get; internal set; }

            public float[,] DInputMag_dOutputAngle { get; internal set; }

            public float[,] DInputAngle_dOutputMag { get; internal set; }

            public float[,] DInputAngle_dOutputAngle { get; internal set; }

            public float[,] DInput2Mag_dOutputMag { get; internal set; }

            public float[,] DInput2Mag_dOutputAngle { get; internal set; }

            public float[,] DInput2Angle_dOutputMag { get; internal set; }

            public float[,] DInput2Angle_dOutputAngle { get; internal set; }

            public float[,] DWeight_dOutputMag { get; internal set; }

            public float[,] DWeight_dOutputAngle { get; internal set; }
        }
    }
}
