//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorDecompositionOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector projection operation.
    /// </summary>
    public class ElementwiseVectorDecompositionOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private CalculatedValues[,] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorDecompositionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector projection function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector projection operation.</param>
        /// <param name="input2">The second input to the element-wise vector projection operation.</param>
        /// <param name="weights">The weights input to the element-wise vector projection operation.</param>
        /// <returns>The output of the element-wise vector projection operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(this.input1.Rows, this.input1.Cols * 10);

            this.calculatedValues = new CalculatedValues[this.input1.Rows, this.input1.Cols / 2];

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    float magnitude = input1[i, j];
                    float angle = input1[i, j + (input1.Cols / 2)];

                    float wMagnitudePivot = input2[i, j * 5];
                    float wAnglePivot = input2[i, (j * 5) + (input2.Cols / 2)];

                    float wMagnitude1 = input2[i, (j * 5) + 1];
                    float wAngle1 = input2[i, (j * 5) + 1 + (input2.Cols / 2)];

                    float wMagnitude2 = input2[i, (j * 5) + 2];
                    float wAngle2 = input2[i, (j * 5) + 2 + (input2.Cols / 2)];

                    float wMagnitude3 = input2[i, (j * 5) + 3];
                    float wAngle3 = input2[i, (j * 5) + 3 + (input2.Cols / 2)];

                    float wMagnitude4 = input2[i, (j * 5) + 4];
                    float wAngle4 = input2[i, (j * 5) + 4 + (input2.Cols / 2)];

                    // Compute vector components
                    float x = magnitude * PradMath.Cos(angle);
                    float y = magnitude * PradMath.Sin(angle);
                    float xPivot = wMagnitudePivot * PradMath.Cos(wAnglePivot);
                    float yPivot = wMagnitudePivot * PradMath.Sin(wAnglePivot);

                    float dx_dMagnitude = PradMath.Cos(angle);
                    float dx_dAngle = -magnitude * PradMath.Sin(angle);
                    float dy_dMagnitude = PradMath.Sin(angle);
                    float dy_dAngle = magnitude * PradMath.Cos(angle);
                    float dXPivot_dWMagnitudePivot = PradMath.Cos(wAnglePivot);
                    float dXPivot_dWAnglePivot = -wMagnitudePivot * PradMath.Sin(wAnglePivot);
                    float dYPivot_dWMagnitudePivot = PradMath.Sin(wAnglePivot);
                    float dYPivot_dWAnglePivot = wMagnitudePivot * PradMath.Cos(wAnglePivot);

                    this.calculatedValues[i, j] = new CalculatedValues()
                    {
                        CV_dx_dMagnitude = dx_dMagnitude,
                        CV_dx_dAngle = dx_dAngle,
                        CV_dy_dMagnitude = dy_dMagnitude,
                        CV_dy_dAngle = dy_dAngle,
                        CV_dXPivot_dWMagnitudePivot = dXPivot_dWMagnitudePivot,
                        CV_dXPivot_dWAnglePivot = dXPivot_dWAnglePivot,
                        CV_dYPivot_dWMagnitudePivot = dYPivot_dWMagnitudePivot,
                        CV_dYPivot_dWAnglePivot = dYPivot_dWAnglePivot,
                    };

                    float x1 = wMagnitude1 * PradMath.Cos(wAngle1);
                    float y1 = wMagnitude1 * PradMath.Sin(wAngle1);

                    float dX1_wMagnitude1 = PradMath.Cos(wAngle1);
                    float dX1_wAngle1 = -wMagnitude1 * PradMath.Sin(wAngle1);
                    float dY1_wMagnitude1 = PradMath.Sin(wAngle1);
                    float dY1_wAngle1 = wMagnitude1 * PradMath.Cos(wAngle1);

                    this.calculatedValues[i, j].CV_dX1_wMagnitude1 = dX1_wMagnitude1;
                    this.calculatedValues[i, j].CV_dX1_wAngle1 = dX1_wAngle1;
                    this.calculatedValues[i, j].CV_dY1_wMagnitude1 = dY1_wMagnitude1;
                    this.calculatedValues[i, j].CV_dY1_wAngle1 = dY1_wAngle1;

                    float x2 = wMagnitude2 * PradMath.Cos(wAngle2);
                    float y2 = wMagnitude2 * PradMath.Sin(wAngle2);

                    float dX2_wMagnitude2 = PradMath.Cos(wAngle2);
                    float dX2_wAngle2 = -wMagnitude2 * PradMath.Sin(wAngle2);
                    float dY2_wMagnitude2 = PradMath.Sin(wAngle2);
                    float dY2_wAngle2 = wMagnitude2 * PradMath.Cos(wAngle2);

                    this.calculatedValues[i, j].CV_dX2_wMagnitude2 = dX2_wMagnitude2;
                    this.calculatedValues[i, j].CV_dX2_wAngle2 = dX2_wAngle2;
                    this.calculatedValues[i, j].CV_dY2_wMagnitude2 = dY2_wMagnitude2;
                    this.calculatedValues[i, j].CV_dY2_wAngle2 = dY2_wAngle2;

                    float x3 = wMagnitude3 * PradMath.Cos(wAngle3);
                    float y3 = wMagnitude3 * PradMath.Sin(wAngle3);

                    float dX3_wMagnitude3 = PradMath.Cos(wAngle3);
                    float dX3_wAngle3 = -wMagnitude3 * PradMath.Sin(wAngle3);
                    float dY3_wMagnitude3 = PradMath.Sin(wAngle3);
                    float dY3_wAngle3 = wMagnitude3 * PradMath.Cos(wAngle3);

                    this.calculatedValues[i, j].CV_dX3_wMagnitude3 = dX3_wMagnitude3;
                    this.calculatedValues[i, j].CV_dX3_wAngle3 = dX3_wAngle3;
                    this.calculatedValues[i, j].CV_dY3_wMagnitude3 = dY3_wMagnitude3;
                    this.calculatedValues[i, j].CV_dY3_wAngle3 = dY3_wAngle3;

                    float x4 = wMagnitude4 * PradMath.Cos(wAngle4);
                    float y4 = wMagnitude4 * PradMath.Sin(wAngle4);

                    float dX4_wMagnitude4 = PradMath.Cos(wAngle4);
                    float dX4_wAngle4 = -wMagnitude4 * PradMath.Sin(wAngle4);
                    float dY4_wMagnitude4 = PradMath.Sin(wAngle4);
                    float dY4_wAngle4 = wMagnitude4 * PradMath.Cos(wAngle4);

                    this.calculatedValues[i, j].CV_dX4_wMagnitude4 = dX4_wMagnitude4;
                    this.calculatedValues[i, j].CV_dX4_wAngle4 = dX4_wAngle4;
                    this.calculatedValues[i, j].CV_dY4_wMagnitude4 = dY4_wMagnitude4;
                    this.calculatedValues[i, j].CV_dY4_wAngle4 = dY4_wAngle4;

                    float sumx = (x + xPivot) / (this.weights[i, j] + 1E-9f);
                    float sumy = (y + yPivot) / (this.weights[i, j] + 1E-9f);

                    float dsumx_dX = 1f / (this.weights[i, j] + 1E-9f);
                    float dsumx_dXPivot = 1f / (this.weights[i, j] + 1E-9f);
                    float dsumx_dWeight = -(x + xPivot) / ((this.weights[i, j] + 1E-9f) * (this.weights[i, j] + 1E-9f));
                    float dsumy_dY = 1f / (this.weights[i, j] + 1E-9f);
                    float dsumy_dYPivot = 1f / (this.weights[i, j] + 1E-9f);
                    float dsumy_dWeight = -(y + yPivot) / ((this.weights[i, j] + 1E-9f) * (this.weights[i, j] + 1E-9f));

                    this.calculatedValues[i, j].CV_dsumx_dX = dsumx_dX;
                    this.calculatedValues[i, j].CV_dsumx_dXPivot = dsumx_dXPivot;
                    this.calculatedValues[i, j].CV_dsumx_dWeight = dsumx_dWeight;
                    this.calculatedValues[i, j].CV_dsumy_dY = dsumy_dY;
                    this.calculatedValues[i, j].CV_dsumy_dYPivot = dsumy_dYPivot;
                    this.calculatedValues[i, j].CV_dsumy_dWeight = dsumy_dWeight;

                    float diffx1 = sumx - x1;
                    float diffy1 = sumy - y1;

                    float dDiffX1_dSumX = 1f;
                    float dDiffX1_dX1 = -1f;
                    float dDiffY1_dSumY = 1f;
                    float dDiffY1_dY1 = -1f;

                    this.calculatedValues[i, j].CV_dDiffX1_dSumX = dDiffX1_dSumX;
                    this.calculatedValues[i, j].CV_dDiffX1_dX1 = dDiffX1_dX1;
                    this.calculatedValues[i, j].CV_dDiffY1_dSumY = dDiffY1_dSumY;
                    this.calculatedValues[i, j].CV_dDiffY1_dY1 = dDiffY1_dY1;

                    float diffx2 = -sumx - x2;
                    float diffy2 = -sumy - y2;

                    float dDiffX2_dSumX = -1f;
                    float dDiffX2_dX2 = -1f;
                    float dDiffY2_dSumY = -1f;
                    float dDiffY2_dY2 = -1f;

                    this.calculatedValues[i, j].CV_dDiffX2_dSumX = dDiffX2_dSumX;
                    this.calculatedValues[i, j].CV_dDiffX2_dX2 = dDiffX2_dX2;
                    this.calculatedValues[i, j].CV_dDiffY2_dSumY = dDiffY2_dSumY;
                    this.calculatedValues[i, j].CV_dDiffY2_dY2 = dDiffY2_dY2;

                    float diffx3 = sumx - x3;
                    float diffy3 = sumy - y3;

                    float dDiffX3_dSumX = 1f;
                    float dDiffX3_dX3 = -1f;
                    float dDiffY3_dSumY = 1f;
                    float dDiffY3_dY3 = -1f;

                    this.calculatedValues[i, j].CV_dDiffX3_dSumX = dDiffX3_dSumX;
                    this.calculatedValues[i, j].CV_dDiffX3_dX3 = dDiffX3_dX3;
                    this.calculatedValues[i, j].CV_dDiffY3_dSumY = dDiffY3_dSumY;
                    this.calculatedValues[i, j].CV_dDiffY3_dY3 = dDiffY3_dY3;

                    float diffx4 = -sumx - x4;
                    float diffy4 = -sumy - y4;

                    float dDiffX4_dSumX = -1f;
                    float dDiffX4_dX4 = -1f;
                    float dDiffY4_dSumY = -1f;
                    float dDiffY4_dY4 = -1f;

                    this.calculatedValues[i, j].CV_dDiffX4_dSumX = dDiffX4_dSumX;
                    this.calculatedValues[i, j].CV_dDiffX4_dX4 = dDiffX4_dX4;
                    this.calculatedValues[i, j].CV_dDiffY4_dSumY = dDiffY4_dSumY;
                    this.calculatedValues[i, j].CV_dDiffY4_dY4 = dDiffY4_dY4;

                    // Compute resultant vector magnitude and angle
                    float resultMagnitude1 = PradMath.Sqrt((diffx1 * diffx1) + (diffy1 * diffy1));
                    float resultAngle1 = PradMath.Atan2(diffy1, diffx1);

                    float dResultMagnitude1_dDiffX1 = diffx1 / resultMagnitude1;
                    float dResultMagnitude1_dDiffY1 = diffy1 / resultMagnitude1;
                    float dResultAngle1_dDiffX1 = -diffy1 / ((diffx1 * diffx1) + (diffy1 * diffy1));
                    float dResultAngle1_dDiffY1 = diffx1 / ((diffx1 * diffx1) + (diffy1 * diffy1));

                    this.calculatedValues[i, j].CV_dResultMagnitude1_dDiffX1 = dResultMagnitude1_dDiffX1;
                    this.calculatedValues[i, j].CV_dResultMagnitude1_dDiffY1 = dResultMagnitude1_dDiffY1;
                    this.calculatedValues[i, j].CV_dResultAngle1_dDiffX1 = dResultAngle1_dDiffX1;
                    this.calculatedValues[i, j].CV_dResultAngle1_dDiffY1 = dResultAngle1_dDiffY1;

                    float resultMagnitude2 = PradMath.Sqrt((diffx2 * diffx2) + (diffy2 * diffy2));
                    float resultAngle2 = PradMath.Atan2(diffy2, diffx2);

                    float dResultMagnitude2_dDiffX2 = diffx2 / resultMagnitude2;
                    float dResultMagnitude2_dDiffY2 = diffy2 / resultMagnitude2;
                    float dResultAngle2_dDiffX2 = -diffy2 / ((diffx2 * diffx2) + (diffy2 * diffy2));
                    float dResultAngle2_dDiffY2 = diffx2 / ((diffx2 * diffx2) + (diffy2 * diffy2));

                    this.calculatedValues[i, j].CV_dResultMagnitude2_dDiffX2 = dResultMagnitude2_dDiffX2;
                    this.calculatedValues[i, j].CV_dResultMagnitude2_dDiffY2 = dResultMagnitude2_dDiffY2;
                    this.calculatedValues[i, j].CV_dResultAngle2_dDiffX2 = dResultAngle2_dDiffX2;
                    this.calculatedValues[i, j].CV_dResultAngle2_dDiffY2 = dResultAngle2_dDiffY2;

                    float resultMagnitude3 = PradMath.Sqrt((diffx3 * diffx3) + (diffy3 * diffy3));
                    float resultAngle3 = PradMath.Atan2(diffy3, diffx3);

                    float dResultMagnitude3_dDiffX3 = diffx3 / resultMagnitude3;
                    float dResultMagnitude3_dDiffY3 = diffy3 / resultMagnitude3;
                    float dResultAngle3_dDiffX3 = -diffy3 / ((diffx3 * diffx3) + (diffy3 * diffy3));
                    float dResultAngle3_dDiffY3 = diffx3 / ((diffx3 * diffx3) + (diffy3 * diffy3));

                    this.calculatedValues[i, j].CV_dResultMagnitude3_dDiffX3 = dResultMagnitude3_dDiffX3;
                    this.calculatedValues[i, j].CV_dResultMagnitude3_dDiffY3 = dResultMagnitude3_dDiffY3;
                    this.calculatedValues[i, j].CV_dResultAngle3_dDiffX3 = dResultAngle3_dDiffX3;
                    this.calculatedValues[i, j].CV_dResultAngle3_dDiffY3 = dResultAngle3_dDiffY3;

                    float resultMagnitude4 = PradMath.Sqrt((diffx4 * diffx4) + (diffy4 * diffy4));
                    float resultAngle4 = PradMath.Atan2(diffy4, diffx4);

                    float dResultMagnitude4_dDiffX4 = diffx4 / resultMagnitude4;
                    float dResultMagnitude4_dDiffY4 = diffy4 / resultMagnitude4;
                    float dResultAngle4_dDiffX4 = -diffy4 / ((diffx4 * diffx4) + (diffy4 * diffy4));
                    float dResultAngle4_dDiffY4 = diffx4 / ((diffx4 * diffx4) + (diffy4 * diffy4));

                    this.calculatedValues[i, j].CV_dResultMagnitude4_dDiffX4 = dResultMagnitude4_dDiffX4;
                    this.calculatedValues[i, j].CV_dResultMagnitude4_dDiffY4 = dResultMagnitude4_dDiffY4;
                    this.calculatedValues[i, j].CV_dResultAngle4_dDiffX4 = dResultAngle4_dDiffX4;
                    this.calculatedValues[i, j].CV_dResultAngle4_dDiffY4 = dResultAngle4_dDiffY4;

                    this.Output[i, j * 10] = magnitude;
                    this.Output[i, (j * 10) + (this.input1.Cols * 10 / 2)] = angle;

                    this.Output[i, (j * 10) + 1] = wMagnitudePivot;
                    this.Output[i, (j * 10) + 1 + (this.input1.Cols * 10 / 2)] = wAnglePivot;

                    this.Output[i, (j * 10) + 2] = wMagnitude1;
                    this.Output[i, (j * 10) + 2 + (this.input1.Cols * 10 / 2)] = wAngle1;

                    this.Output[i, (j * 10) + 3] = wMagnitude2;
                    this.Output[i, (j * 10) + 3 + (this.input1.Cols * 10 / 2)] = wAngle2;

                    this.Output[i, (j * 10) + 4] = wMagnitude3;
                    this.Output[i, (j * 10) + 4 + (this.input1.Cols * 10 / 2)] = wAngle3;

                    this.Output[i, (j * 10) + 5] = wMagnitude4;
                    this.Output[i, (j * 10) + 5 + (this.input1.Cols * 10 / 2)] = wAngle4;

                    this.Output[i, (j * 10) + 6] = resultMagnitude1;
                    this.Output[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] = resultAngle1;

                    this.Output[i, (j * 10) + 7] = resultMagnitude2;
                    this.Output[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] = resultAngle2;

                    this.Output[i, (j * 10) + 8] = resultMagnitude3;
                    this.Output[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] = resultAngle3;

                    this.Output[i, (j * 10) + 9] = resultMagnitude4;
                    this.Output[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] = resultAngle4;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    var grad = this.calculatedValues[i, j];

                    dInput1[i, j] += dOutput[i, j * 10];
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + (this.input1.Cols * 10 / 2)];

                    dInput1[i, j] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 6] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;

                    dInput1[i, j] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 7] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;

                    dInput1[i, j] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 8] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;

                    dInput1[i, j] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 9] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;

                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 1];
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 1 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, (j * 5) + 1] += dOutput[i, (j * 10) + 2];
                    dInput2[i, (j * 5) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 2 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, (j * 5) + 2] += dOutput[i, (j * 10) + 3];
                    dInput2[i, (j * 5) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 3 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, (j * 5) + 3] += dOutput[i, (j * 10) + 4];
                    dInput2[i, (j * 5) + 3 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 4 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, (j * 5) + 4] += dOutput[i, (j * 10) + 5];
                    dInput2[i, (j * 5) + 4 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 5 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;

                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;

                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;

                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;

                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;

                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;

                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;

                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;
                    dInput2[i, (j * 5) + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;

                    dInput2[i, (j * 5) + 1] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wMagnitude1;
                    dInput2[i, (j * 5) + 1] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wMagnitude1;
                    dInput2[i, (j * 5) + 1] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wMagnitude1;
                    dInput2[i, (j * 5) + 1] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wMagnitude1;

                    dInput2[i, (j * 5) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wAngle1;
                    dInput2[i, (j * 5) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wAngle1;
                    dInput2[i, (j * 5) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wAngle1;
                    dInput2[i, (j * 5) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wAngle1;

                    dInput2[i, (j * 5) + 2] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wMagnitude2;
                    dInput2[i, (j * 5) + 2] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wMagnitude2;
                    dInput2[i, (j * 5) + 2] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wMagnitude2;
                    dInput2[i, (j * 5) + 2] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wMagnitude2;

                    dInput2[i, (j * 5) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wAngle2;
                    dInput2[i, (j * 5) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wAngle2;
                    dInput2[i, (j * 5) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wAngle2;
                    dInput2[i, (j * 5) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wAngle2;

                    dInput2[i, (j * 5) + 3] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dX3 * grad.CV_dX3_wMagnitude3;
                    dInput2[i, (j * 5) + 3] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dY3 * grad.CV_dY3_wMagnitude3;
                    dInput2[i, (j * 5) + 3] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dX3 * grad.CV_dX3_wMagnitude3;
                    dInput2[i, (j * 5) + 3] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dY3 * grad.CV_dY3_wMagnitude3;

                    dInput2[i, (j * 5) + 3 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dX3 * grad.CV_dX3_wAngle3;
                    dInput2[i, (j * 5) + 3 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dX3 * grad.CV_dX3_wAngle3;
                    dInput2[i, (j * 5) + 3 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dY3 * grad.CV_dY3_wAngle3;
                    dInput2[i, (j * 5) + 3 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dY3 * grad.CV_dY3_wAngle3;

                    dInput2[i, (j * 5) + 4] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dX4 * grad.CV_dX4_wMagnitude4;
                    dInput2[i, (j * 5) + 4] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dY4 * grad.CV_dY4_wMagnitude4;
                    dInput2[i, (j * 5) + 4] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dX4 * grad.CV_dX4_wMagnitude4;
                    dInput2[i, (j * 5) + 4] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dY4 * grad.CV_dY4_wMagnitude4;

                    dInput2[i, (j * 5) + 4 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dX4 * grad.CV_dX4_wAngle4;
                    dInput2[i, (j * 5) + 4 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dX4 * grad.CV_dX4_wAngle4;
                    dInput2[i, (j * 5) + 4 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dY4 * grad.CV_dY4_wAngle4;
                    dInput2[i, (j * 5) + 4 + (this.input2.Cols / 2)] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dY4 * grad.CV_dY4_wAngle4;

                    dWeights[i, j] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 6] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 7] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 8] * grad.CV_dResultMagnitude3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffX3 * grad.CV_dDiffX3_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle3_dDiffY3 * grad.CV_dDiffY3_dSumY * grad.CV_dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 9] * grad.CV_dResultMagnitude4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffX4 * grad.CV_dDiffX4_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] * grad.CV_dResultAngle4_dDiffY4 * grad.CV_dDiffY4_dSumY * grad.CV_dsumy_dWeight;
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
            public float CV_dx_dMagnitude { get; internal set; }

            public float CV_dx_dAngle { get; internal set; }

            public float CV_dy_dMagnitude { get; internal set; }

            public float CV_dy_dAngle { get; internal set; }

            public float CV_dXPivot_dWMagnitudePivot { get; internal set; }

            public float CV_dXPivot_dWAnglePivot { get; internal set; }

            public float CV_dYPivot_dWMagnitudePivot { get; internal set; }

            public float CV_dYPivot_dWAnglePivot { get; internal set; }

            public float CV_dX1_wMagnitude1 { get; internal set; }

            public float CV_dX1_wAngle1 { get; internal set; }

            public float CV_dY1_wMagnitude1 { get; internal set; }

            public float CV_dY1_wAngle1 { get; internal set; }

            public float CV_dX2_wMagnitude2 { get; internal set; }

            public float CV_dX2_wAngle2 { get; internal set; }

            public float CV_dY2_wMagnitude2 { get; internal set; }

            public float CV_dY2_wAngle2 { get; internal set; }

            public float CV_dX3_wMagnitude3 { get; internal set; }

            public float CV_dX3_wAngle3 { get; internal set; }

            public float CV_dY3_wMagnitude3 { get; internal set; }

            public float CV_dY3_wAngle3 { get; internal set; }

            public float CV_dX4_wMagnitude4 { get; internal set; }

            public float CV_dX4_wAngle4 { get; internal set; }

            public float CV_dY4_wMagnitude4 { get; internal set; }

            public float CV_dY4_wAngle4 { get; internal set; }

            public float CV_dsumx_dX { get; internal set; }

            public float CV_dsumx_dXPivot { get; internal set; }

            public float CV_dsumx_dWeight { get; internal set; }

            public float CV_dsumy_dY { get; internal set; }

            public float CV_dsumy_dYPivot { get; internal set; }

            public float CV_dsumy_dWeight { get; internal set; }

            public float CV_dDiffX1_dSumX { get; internal set; }

            public float CV_dDiffX1_dX1 { get; internal set; }

            public float CV_dDiffY1_dSumY { get; internal set; }

            public float CV_dDiffY1_dY1 { get; internal set; }

            public float CV_dDiffX2_dSumX { get; internal set; }

            public float CV_dDiffX2_dX2 { get; internal set; }

            public float CV_dDiffY2_dSumY { get; internal set; }

            public float CV_dDiffY2_dY2 { get; internal set; }

            public float CV_dDiffX3_dSumX { get; internal set; }

            public float CV_dDiffX3_dX3 { get; internal set; }

            public float CV_dDiffY3_dSumY { get; internal set; }

            public float CV_dDiffY3_dY3 { get; internal set; }

            public float CV_dDiffX4_dSumX { get; internal set; }

            public float CV_dDiffX4_dX4 { get; internal set; }

            public float CV_dDiffY4_dSumY { get; internal set; }

            public float CV_dDiffY4_dY4 { get; internal set; }

            public float CV_dResultMagnitude1_dDiffX1 { get; internal set; }

            public float CV_dResultMagnitude1_dDiffY1 { get; internal set; }

            public float CV_dResultAngle1_dDiffX1 { get; internal set; }

            public float CV_dResultAngle1_dDiffY1 { get; internal set; }

            public float CV_dResultMagnitude2_dDiffX2 { get; internal set; }

            public float CV_dResultMagnitude2_dDiffY2 { get; internal set; }

            public float CV_dResultAngle2_dDiffX2 { get; internal set; }

            public float CV_dResultAngle2_dDiffY2 { get; internal set; }

            public float CV_dResultMagnitude3_dDiffX3 { get; internal set; }

            public float CV_dResultMagnitude3_dDiffY3 { get; internal set; }

            public float CV_dResultAngle3_dDiffX3 { get; internal set; }

            public float CV_dResultAngle3_dDiffY3 { get; internal set; }

            public float CV_dResultMagnitude4_dDiffX4 { get; internal set; }

            public float CV_dResultMagnitude4_dDiffY4 { get; internal set; }

            public float CV_dResultAngle4_dDiffX4 { get; internal set; }

            public float CV_dResultAngle4_dDiffY4 { get; internal set; }
        }
    }
}
