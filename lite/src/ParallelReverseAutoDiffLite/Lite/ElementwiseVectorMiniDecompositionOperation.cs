//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorMiniDecompositionOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector mini decomposition operation.
    /// </summary>
    public class ElementwiseVectorMiniDecompositionOperation : Operation
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
            return new ElementwiseVectorMiniDecompositionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector mini decomposition function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector mini decomposition operation.</param>
        /// <param name="input2">The second input to the element-wise vector mini decomposition operation.</param>
        /// <param name="weights">The weights input to the element-wise vector mini decomposition operation.</param>
        /// <returns>The output of the element-wise vector mini decomposition operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(this.input1.Rows, this.input1.Cols * 6);

            this.calculatedValues = new CalculatedValues[this.input1.Rows, this.input1.Cols / 2];

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    float magnitude = input1[i, j];
                    float angle = input1[i, j + (input1.Cols / 2)];

                    float wMagnitudePivot = input2[i, j * 3];
                    float wAnglePivot = input2[i, (j * 3) + (input2.Cols / 2)];

                    float wMagnitude1 = input2[i, (j * 3) + 1];
                    float wAngle1 = input2[i, (j * 3) + 1 + (input2.Cols / 2)];

                    float wMagnitude2 = input2[i, (j * 3) + 2];
                    float wAngle2 = input2[i, (j * 3) + 2 + (input2.Cols / 2)];

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

                    this.Output[i, j * 6] = magnitude;
                    this.Output[i, (j * 6) + (this.input1.Cols * 6 / 2)] = angle;

                    this.Output[i, (j * 6) + 1] = wMagnitudePivot;
                    this.Output[i, (j * 6) + 1 + (this.input1.Cols * 6 / 2)] = wAnglePivot;

                    this.Output[i, (j * 6) + 2] = wMagnitude1;
                    this.Output[i, (j * 6) + 2 + (this.input1.Cols * 6 / 2)] = wAngle1;

                    this.Output[i, (j * 6) + 3] = wMagnitude2;
                    this.Output[i, (j * 6) + 3 + (this.input1.Cols * 6 / 2)] = wAngle2;

                    this.Output[i, (j * 6) + 4] = resultMagnitude1;
                    this.Output[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] = resultAngle1;

                    this.Output[i, (j * 6) + 5] = resultMagnitude2;
                    this.Output[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] = resultAngle2;
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

                    dInput1[i, j] += dOutput[i, j * 6];
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + (this.input1.Cols * 6 / 2)];

                    dInput1[i, j] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 4] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;

                    dInput1[i, j] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 5] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dX * grad.CV_dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dY * grad.CV_dy_dAngle;

                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 1];
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 1 + (this.input1.Cols * 6 / 2)];

                    dInput2[i, (j * 3) + 1] += dOutput[i, (j * 6) + 2];
                    dInput2[i, (j * 3) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 2 + (this.input1.Cols * 6 / 2)];

                    dInput2[i, (j * 3) + 2] += dOutput[i, (j * 6) + 3];
                    dInput2[i, (j * 3) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 3 + (this.input1.Cols * 6 / 2)];

                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;

                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;

                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 3] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWMagnitudePivot;

                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dXPivot * grad.CV_dXPivot_dWAnglePivot;
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;
                    dInput2[i, (j * 3) + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dYPivot * grad.CV_dYPivot_dWAnglePivot;

                    dInput2[i, (j * 3) + 1] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wMagnitude1;
                    dInput2[i, (j * 3) + 1] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wMagnitude1;
                    dInput2[i, (j * 3) + 1] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wMagnitude1;
                    dInput2[i, (j * 3) + 1] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wMagnitude1;

                    dInput2[i, (j * 3) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wAngle1;
                    dInput2[i, (j * 3) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dX1 * grad.CV_dX1_wAngle1;
                    dInput2[i, (j * 3) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wAngle1;
                    dInput2[i, (j * 3) + 1 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dY1 * grad.CV_dY1_wAngle1;

                    dInput2[i, (j * 3) + 2] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wMagnitude2;
                    dInput2[i, (j * 3) + 2] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wMagnitude2;
                    dInput2[i, (j * 3) + 2] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wMagnitude2;
                    dInput2[i, (j * 3) + 2] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wMagnitude2;

                    dInput2[i, (j * 3) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wAngle2;
                    dInput2[i, (j * 3) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dX2 * grad.CV_dX2_wAngle2;
                    dInput2[i, (j * 3) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wAngle2;
                    dInput2[i, (j * 3) + 2 + (this.input2.Cols / 2)] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dY2 * grad.CV_dY2_wAngle2;

                    dWeights[i, j] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 6) + 4] * grad.CV_dResultMagnitude1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffX1 * grad.CV_dDiffX1_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 6) + 4 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle1_dDiffY1 * grad.CV_dDiffY1_dSumY * grad.CV_dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 6) + 5] * grad.CV_dResultMagnitude2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffX2 * grad.CV_dDiffX2_dSumX * grad.CV_dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, (j * 6) + 5 + (this.input1.Cols * 6 / 2)] * grad.CV_dResultAngle2_dDiffY2 * grad.CV_dDiffY2_dSumY * grad.CV_dsumy_dWeight;
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

            public float CV_dResultMagnitude1_dDiffX1 { get; internal set; }

            public float CV_dResultMagnitude1_dDiffY1 { get; internal set; }

            public float CV_dResultAngle1_dDiffX1 { get; internal set; }

            public float CV_dResultAngle1_dDiffY1 { get; internal set; }

            public float CV_dResultMagnitude2_dDiffX2 { get; internal set; }

            public float CV_dResultMagnitude2_dDiffY2 { get; internal set; }

            public float CV_dResultAngle2_dDiffX2 { get; internal set; }

            public float CV_dResultAngle2_dDiffY2 { get; internal set; }
        }
    }
}
