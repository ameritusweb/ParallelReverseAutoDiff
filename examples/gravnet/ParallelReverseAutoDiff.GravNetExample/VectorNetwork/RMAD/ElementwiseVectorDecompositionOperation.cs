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
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];

                    double wMagnitudePivot = input2[i, j * 5];
                    double wAnglePivot = input2[i, j * 5 + (input2.Cols / 2)];

                    double wMagnitude1 = input2[i, j * 5 + 1];
                    double wAngle1 = input2[i, j * 5 + 1 + (input2.Cols / 2)];

                    double wMagnitude2 = input2[i, j * 5 + 2];
                    double wAngle2 = input2[i, j * 5 + 2 + (input2.Cols / 2)];

                    double wMagnitude3 = input2[i, j * 5 + 3];
                    double wAngle3 = input2[i, j * 5 + 3 + (input2.Cols / 2)];

                    double wMagnitude4 = input2[i, j * 5 + 4];
                    double wAngle4 = input2[i, j * 5 + 4 + (input2.Cols / 2)];

                    // Compute vector components
                    double x = magnitude * Math.Cos(angle);
                    double y = magnitude * Math.Sin(angle);
                    double xPivot = wMagnitudePivot * Math.Cos(wAnglePivot);
                    double yPivot = wMagnitudePivot * Math.Sin(wAnglePivot);

                    double dx_dMagnitude = Math.Cos(angle);
                    double dx_dAngle = -magnitude * Math.Sin(angle);
                    double dy_dMagnitude = Math.Sin(angle);
                    double dy_dAngle = magnitude * Math.Cos(angle);
                    double dXPivot_dWMagnitudePivot = Math.Cos(wAnglePivot);
                    double dXPivot_dWAnglePivot = -wMagnitudePivot * Math.Sin(wAnglePivot);
                    double dYPivot_dWMagnitudePivot = Math.Sin(wAnglePivot);
                    double dYPivot_dWAnglePivot = wMagnitudePivot * Math.Cos(wAnglePivot);

                    this.calculatedValues[i, j] = new CalculatedValues()
                    {
                        dx_dMagnitude = dx_dMagnitude,
                        dx_dAngle = dx_dAngle,
                        dy_dMagnitude = dy_dMagnitude,
                        dy_dAngle = dy_dAngle,
                        dXPivot_dWMagnitudePivot = dXPivot_dWMagnitudePivot,
                        dXPivot_dWAnglePivot = dXPivot_dWAnglePivot,
                        dYPivot_dWMagnitudePivot = dYPivot_dWMagnitudePivot,
                        dYPivot_dWAnglePivot = dYPivot_dWAnglePivot,
                    };

                    double x1 = wMagnitude1 * Math.Cos(wAngle1);
                    double y1 = wMagnitude1 * Math.Sin(wAngle1);

                    double dX1_wMagnitude1 = Math.Cos(wAngle1);
                    double dX1_wAngle1 = -wMagnitude1 * Math.Sin(wAngle1);
                    double dY1_wMagnitude1 = Math.Sin(wAngle1);
                    double dY1_wAngle1 = wMagnitude1 * Math.Cos(wAngle1);

                    this.calculatedValues[i, j].dX1_wMagnitude1 = dX1_wMagnitude1;
                    this.calculatedValues[i, j].dX1_wAngle1 = dX1_wAngle1;
                    this.calculatedValues[i, j].dY1_wMagnitude1 = dY1_wMagnitude1;
                    this.calculatedValues[i, j].dY1_wAngle1 = dY1_wAngle1;

                    double x2 = wMagnitude2 * Math.Cos(wAngle2);
                    double y2 = wMagnitude2 * Math.Sin(wAngle2);

                    double dX2_wMagnitude2 = Math.Cos(wAngle2);
                    double dX2_wAngle2 = -wMagnitude2 * Math.Sin(wAngle2);
                    double dY2_wMagnitude2 = Math.Sin(wAngle2);
                    double dY2_wAngle2 = wMagnitude2 * Math.Cos(wAngle2);

                    this.calculatedValues[i, j].dX2_wMagnitude2 = dX2_wMagnitude2;
                    this.calculatedValues[i, j].dX2_wAngle2 = dX2_wAngle2;
                    this.calculatedValues[i, j].dY2_wMagnitude2 = dY2_wMagnitude2;
                    this.calculatedValues[i, j].dY2_wAngle2 = dY2_wAngle2;

                    double x3 = wMagnitude3 * Math.Cos(wAngle3);
                    double y3 = wMagnitude3 * Math.Sin(wAngle3);

                    double dX3_wMagnitude3 = Math.Cos(wAngle3);
                    double dX3_wAngle3 = -wMagnitude3 * Math.Sin(wAngle3);
                    double dY3_wMagnitude3 = Math.Sin(wAngle3);
                    double dY3_wAngle3 = wMagnitude3 * Math.Cos(wAngle3);

                    this.calculatedValues[i, j].dX3_wMagnitude3 = dX3_wMagnitude3;
                    this.calculatedValues[i, j].dX3_wAngle3 = dX3_wAngle3;
                    this.calculatedValues[i, j].dY3_wMagnitude3 = dY3_wMagnitude3;
                    this.calculatedValues[i, j].dY3_wAngle3 = dY3_wAngle3;

                    double x4 = wMagnitude4 * Math.Cos(wAngle4);
                    double y4 = wMagnitude4 * Math.Sin(wAngle4);

                    double dX4_wMagnitude4 = Math.Cos(wAngle4);
                    double dX4_wAngle4 = -wMagnitude4 * Math.Sin(wAngle4);
                    double dY4_wMagnitude4 = Math.Sin(wAngle4);
                    double dY4_wAngle4 = wMagnitude4 * Math.Cos(wAngle4);

                    this.calculatedValues[i, j].dX4_wMagnitude4 = dX4_wMagnitude4;
                    this.calculatedValues[i, j].dX4_wAngle4 = dX4_wAngle4;
                    this.calculatedValues[i, j].dY4_wMagnitude4 = dY4_wMagnitude4;
                    this.calculatedValues[i, j].dY4_wAngle4 = dY4_wAngle4;

                    double sumx = (x + xPivot) / (this.weights[i, j] + 1E-9);
                    double sumy = (y + yPivot) / (this.weights[i, j] + 1E-9);

                    double dsumx_dX = 1d / (this.weights[i, j] + 1E-9);
                    double dsumx_dXPivot = 1d / (this.weights[i, j] + 1E-9);
                    double dsumx_dWeight = -(x + xPivot) / ((this.weights[i, j] + 1E-9) * (this.weights[i, j] + 1E-9));
                    double dsumy_dY = 1d / (this.weights[i, j] + 1E-9);
                    double dsumy_dYPivot = 1d / (this.weights[i, j] + 1E-9);
                    double dsumy_dWeight = -(y + yPivot) / ((this.weights[i, j] + 1E-9) * (this.weights[i, j] + 1E-9));

                    this.calculatedValues[i, j].dsumx_dX = dsumx_dX;
                    this.calculatedValues[i, j].dsumx_dXPivot = dsumx_dXPivot;
                    this.calculatedValues[i, j].dsumx_dWeight = dsumx_dWeight;
                    this.calculatedValues[i, j].dsumy_dY = dsumy_dY;
                    this.calculatedValues[i, j].dsumy_dYPivot = dsumy_dYPivot;
                    this.calculatedValues[i, j].dsumy_dWeight = dsumy_dWeight;

                    double diffx1 = sumx - x1;
                    double diffy1 = sumy - y1;

                    double dDiffX1_dSumX = 1d;
                    double dDiffX1_dX1 = -1d;
                    double dDiffY1_dSumY = 1d;
                    double dDiffY1_dY1 = -1d;

                    this.calculatedValues[i, j].dDiffX1_dSumX = dDiffX1_dSumX;
                    this.calculatedValues[i, j].dDiffX1_dX1 = dDiffX1_dX1;
                    this.calculatedValues[i, j].dDiffY1_dSumY = dDiffY1_dSumY;
                    this.calculatedValues[i, j].dDiffY1_dY1 = dDiffY1_dY1;

                    double diffx2 = -sumx - x2;
                    double diffy2 = -sumy - y2;

                    double dDiffX2_dSumX = -1d;
                    double dDiffX2_dX2 = -1d;
                    double dDiffY2_dSumY = -1d;
                    double dDiffY2_dY2 = -1d;

                    this.calculatedValues[i, j].dDiffX2_dSumX = dDiffX2_dSumX;
                    this.calculatedValues[i, j].dDiffX2_dX2 = dDiffX2_dX2;
                    this.calculatedValues[i, j].dDiffY2_dSumY = dDiffY2_dSumY;
                    this.calculatedValues[i, j].dDiffY2_dY2 = dDiffY2_dY2;

                    double diffx3 = sumx - x3;
                    double diffy3 = sumy - y3;

                    double dDiffX3_dSumX = 1d;
                    double dDiffX3_dX3 = -1d;
                    double dDiffY3_dSumY = 1d;
                    double dDiffY3_dY3 = -1d;

                    this.calculatedValues[i, j].dDiffX3_dSumX = dDiffX3_dSumX;
                    this.calculatedValues[i, j].dDiffX3_dX3 = dDiffX3_dX3;
                    this.calculatedValues[i, j].dDiffY3_dSumY = dDiffY3_dSumY;
                    this.calculatedValues[i, j].dDiffY3_dY3 = dDiffY3_dY3;

                    double diffx4 = -sumx - x4;
                    double diffy4 = -sumy - y4;

                    double dDiffX4_dSumX = -1d;
                    double dDiffX4_dX4 = -1d;
                    double dDiffY4_dSumY = -1d;
                    double dDiffY4_dY4 = -1d;

                    this.calculatedValues[i, j].dDiffX4_dSumX = dDiffX4_dSumX;
                    this.calculatedValues[i, j].dDiffX4_dX4 = dDiffX4_dX4;
                    this.calculatedValues[i, j].dDiffY4_dSumY = dDiffY4_dSumY;
                    this.calculatedValues[i, j].dDiffY4_dY4 = dDiffY4_dY4;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude1 = Math.Sqrt((diffx1 * diffx1) + (diffy1 * diffy1));
                    double resultAngle1 = Math.Atan2(diffy1, diffx1);

                    double dResultMagnitude1_dDiffX1 = diffx1 / resultMagnitude1;
                    double dResultMagnitude1_dDiffY1 = diffy1 / resultMagnitude1;
                    double dResultAngle1_dDiffX1 = -diffy1 / (diffx1 * diffx1 + diffy1 * diffy1);
                    double dResultAngle1_dDiffY1 = diffx1 / (diffx1 * diffx1 + diffy1 * diffy1);

                    this.calculatedValues[i, j].dResultMagnitude1_dDiffX1 = dResultMagnitude1_dDiffX1;
                    this.calculatedValues[i, j].dResultMagnitude1_dDiffY1 = dResultMagnitude1_dDiffY1;
                    this.calculatedValues[i, j].dResultAngle1_dDiffX1 = dResultAngle1_dDiffX1;
                    this.calculatedValues[i, j].dResultAngle1_dDiffY1 = dResultAngle1_dDiffY1;

                    double resultMagnitude2 = Math.Sqrt((diffx2 * diffx2) + (diffy2 * diffy2));
                    double resultAngle2 = Math.Atan2(diffy2, diffx2);

                    double dResultMagnitude2_dDiffX2 = diffx2 / resultMagnitude2;
                    double dResultMagnitude2_dDiffY2 = diffy2 / resultMagnitude2;
                    double dResultAngle2_dDiffX2 = -diffy2 / (diffx2 * diffx2 + diffy2 * diffy2);
                    double dResultAngle2_dDiffY2 = diffx2 / (diffx2 * diffx2 + diffy2 * diffy2);

                    this.calculatedValues[i, j].dResultMagnitude2_dDiffX2 = dResultMagnitude2_dDiffX2;
                    this.calculatedValues[i, j].dResultMagnitude2_dDiffY2 = dResultMagnitude2_dDiffY2;
                    this.calculatedValues[i, j].dResultAngle2_dDiffX2 = dResultAngle2_dDiffX2;
                    this.calculatedValues[i, j].dResultAngle2_dDiffY2 = dResultAngle2_dDiffY2;

                    double resultMagnitude3 = Math.Sqrt((diffx3 * diffx3) + (diffy3 * diffy3));
                    double resultAngle3 = Math.Atan2(diffy3, diffx3);

                    double dResultMagnitude3_dDiffX3 = diffx3 / resultMagnitude3;
                    double dResultMagnitude3_dDiffY3 = diffy3 / resultMagnitude3;
                    double dResultAngle3_dDiffX3 = -diffy3 / (diffx3 * diffx3 + diffy3 * diffy3);
                    double dResultAngle3_dDiffY3 = diffx3 / (diffx3 * diffx3 + diffy3 * diffy3);

                    this.calculatedValues[i, j].dResultMagnitude3_dDiffX3 = dResultMagnitude3_dDiffX3;
                    this.calculatedValues[i, j].dResultMagnitude3_dDiffY3 = dResultMagnitude3_dDiffY3;
                    this.calculatedValues[i, j].dResultAngle3_dDiffX3 = dResultAngle3_dDiffX3;
                    this.calculatedValues[i, j].dResultAngle3_dDiffY3 = dResultAngle3_dDiffY3;

                    double resultMagnitude4 = Math.Sqrt((diffx4 * diffx4) + (diffy4 * diffy4));
                    double resultAngle4 = Math.Atan2(diffy4, diffx4);

                    double dResultMagnitude4_dDiffX4 = diffx4 / resultMagnitude4;
                    double dResultMagnitude4_dDiffY4 = diffy4 / resultMagnitude4;
                    double dResultAngle4_dDiffX4 = -diffy4 / (diffx4 * diffx4 + diffy4 * diffy4);
                    double dResultAngle4_dDiffY4 = diffx4 / (diffx4 * diffx4 + diffy4 * diffy4);

                    this.calculatedValues[i, j].dResultMagnitude4_dDiffX4 = dResultMagnitude4_dDiffX4;
                    this.calculatedValues[i, j].dResultMagnitude4_dDiffY4 = dResultMagnitude4_dDiffY4;
                    this.calculatedValues[i, j].dResultAngle4_dDiffX4 = dResultAngle4_dDiffX4;
                    this.calculatedValues[i, j].dResultAngle4_dDiffY4 = dResultAngle4_dDiffY4;

                    this.Output[i, j * 10] = magnitude;
                    this.Output[i, j * 10 + (this.input1.Cols * 10 / 2)] = angle;

                    this.Output[i, j * 10 + 1] = wMagnitudePivot;
                    this.Output[i, j * 10 + 1 + (this.input1.Cols * 10 / 2)] = wAnglePivot;

                    this.Output[i, j * 10 + 2] = wMagnitude1;
                    this.Output[i, j * 10 + 2 + (this.input1.Cols * 10 / 2)] = wAngle1;

                    this.Output[i, j * 10 + 3] = wMagnitude2;
                    this.Output[i, j * 10 + 3 + (this.input1.Cols * 10 / 2)] = wAngle2;

                    this.Output[i, j * 10 + 4] = wMagnitude3;
                    this.Output[i, j * 10 + 4 + (this.input1.Cols * 10 / 2)] = wAngle3;

                    this.Output[i, j * 10 + 5] = wMagnitude4;
                    this.Output[i, j * 10 + 5 + (this.input1.Cols * 10 / 2)] = wAngle4;

                    this.Output[i, j * 10 + 6] = resultMagnitude1;
                    this.Output[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] = resultAngle1;

                    this.Output[i, j * 10 + 7] = resultMagnitude2;
                    this.Output[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] = resultAngle2;

                    this.Output[i, j * 10 + 8] = resultMagnitude3;
                    this.Output[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] = resultAngle3;

                    this.Output[i, j * 10 + 9] = resultMagnitude4;
                    this.Output[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] = resultAngle4;
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
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + (this.input1.Cols * 10 / 2)];

                    dInput1[i, j] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 6] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dY * grad.dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dY * grad.dy_dAngle;

                    dInput1[i, j] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 7] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dY * grad.dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dY * grad.dy_dAngle;

                    dInput1[i, j] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 8] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dY * grad.dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dY * grad.dy_dAngle;

                    dInput1[i, j] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dX * grad.dx_dMagnitude;
                    dInput1[i, j] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dY * grad.dy_dMagnitude;

                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 9] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dY * grad.dy_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dX * grad.dx_dAngle;
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dY * grad.dy_dAngle;

                    dInput2[i, j * 5] += dOutput[i, j * 10 + 1];
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 1 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, j * 5 + 1] += dOutput[i, j * 10 + 2];
                    dInput2[i, j * 5 + 1 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 2 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, j * 5 + 2] += dOutput[i, j * 10 + 3];
                    dInput2[i, j * 5 + 2 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 3 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, j * 5 + 3] += dOutput[i, j * 10 + 4];
                    dInput2[i, j * 5 + 3 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 4 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, j * 5 + 4] += dOutput[i, j * 10 + 5];
                    dInput2[i, j * 5 + 4 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 5 + (this.input1.Cols * 10 / 2)];

                    dInput2[i, j * 5] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;

                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;

                    dInput2[i, j * 5] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;

                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;

                    dInput2[i, j * 5] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;

                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;

                    dInput2[i, j * 5] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWMagnitudePivot;
                    dInput2[i, j * 5] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWMagnitudePivot;

                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dXPivot * grad.dXPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;
                    dInput2[i, j * 5 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dYPivot * grad.dYPivot_dWAnglePivot;

                    dInput2[i, j * 5 + 1] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dX1 * grad.dX1_wMagnitude1;
                    dInput2[i, j * 5 + 1] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dY1 * grad.dY1_wMagnitude1;
                    dInput2[i, j * 5 + 1] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dX1 * grad.dX1_wMagnitude1;
                    dInput2[i, j * 5 + 1] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dY1 * grad.dY1_wMagnitude1;

                    dInput2[i, j * 5 + 1 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dX1 * grad.dX1_wAngle1;
                    dInput2[i, j * 5 + 1 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dX1 * grad.dX1_wAngle1;
                    dInput2[i, j * 5 + 1 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dY1 * grad.dY1_wAngle1;
                    dInput2[i, j * 5 + 1 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dY1 * grad.dY1_wAngle1;

                    dInput2[i, j * 5 + 2] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dX2 * grad.dX2_wMagnitude2;
                    dInput2[i, j * 5 + 2] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dY2 * grad.dY2_wMagnitude2;
                    dInput2[i, j * 5 + 2] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dX2 * grad.dX2_wMagnitude2;
                    dInput2[i, j * 5 + 2] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dY2 * grad.dY2_wMagnitude2;

                    dInput2[i, j * 5 + 2 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dX2 * grad.dX2_wAngle2;
                    dInput2[i, j * 5 + 2 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dX2 * grad.dX2_wAngle2;
                    dInput2[i, j * 5 + 2 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dY2 * grad.dY2_wAngle2;
                    dInput2[i, j * 5 + 2 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dY2 * grad.dY2_wAngle2;

                    dInput2[i, j * 5 + 3] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dX3 * grad.dX3_wMagnitude3;
                    dInput2[i, j * 5 + 3] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dY3 * grad.dY3_wMagnitude3;
                    dInput2[i, j * 5 + 3] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dX3 * grad.dX3_wMagnitude3;
                    dInput2[i, j * 5 + 3] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dY3 * grad.dY3_wMagnitude3;

                    dInput2[i, j * 5 + 3 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dX3 * grad.dX3_wAngle3;
                    dInput2[i, j * 5 + 3 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dX3 * grad.dX3_wAngle3;
                    dInput2[i, j * 5 + 3 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dY3 * grad.dY3_wAngle3;
                    dInput2[i, j * 5 + 3 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dY3 * grad.dY3_wAngle3;

                    dInput2[i, j * 5 + 4] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dX4 * grad.dX4_wMagnitude4;
                    dInput2[i, j * 5 + 4] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dY4 * grad.dY4_wMagnitude4;
                    dInput2[i, j * 5 + 4] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dX4 * grad.dX4_wMagnitude4;
                    dInput2[i, j * 5 + 4] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dY4 * grad.dY4_wMagnitude4;

                    dInput2[i, j * 5 + 4 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dX4 * grad.dX4_wAngle4;
                    dInput2[i, j * 5 + 4 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dX4 * grad.dX4_wAngle4;
                    dInput2[i, j * 5 + 4 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dY4 * grad.dY4_wAngle4;
                    dInput2[i, j * 5 + 4 + (this.input2.Cols / 2)] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dY4 * grad.dY4_wAngle4;

                    dWeights[i, j] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 6] * grad.dResultMagnitude1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffX1 * grad.dDiffX1_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 6 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle1_dDiffY1 * grad.dDiffY1_dSumY * grad.dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 7] * grad.dResultMagnitude2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffX2 * grad.dDiffX2_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 7 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle2_dDiffY2 * grad.dDiffY2_dSumY * grad.dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 8] * grad.dResultMagnitude3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffX3 * grad.dDiffX3_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 8 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle3_dDiffY3 * grad.dDiffY3_dSumY * grad.dsumy_dWeight;

                    dWeights[i, j] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 9] * grad.dResultMagnitude4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffX4 * grad.dDiffX4_dSumX * grad.dsumx_dWeight;
                    dWeights[i, j] += dOutput[i, j * 10 + 9 + (this.input1.Cols * 10 / 2)] * grad.dResultAngle4_dDiffY4 * grad.dDiffY4_dSumY * grad.dsumy_dWeight;
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
            public double dx_dMagnitude { get; internal set; }
            public double dx_dAngle { get; internal set; }
            public double dy_dMagnitude { get; internal set; }
            public double dy_dAngle { get; internal set; }
            public double dXPivot_dWMagnitudePivot { get; internal set; }
            public double dXPivot_dWAnglePivot { get; internal set; }
            public double dYPivot_dWMagnitudePivot { get; internal set; }
            public double dYPivot_dWAnglePivot { get; internal set; }
            public double dX1_wMagnitude1 { get; internal set; }
            public double dX1_wAngle1 { get; internal set; }
            public double dY1_wMagnitude1 { get; internal set; }
            public double dY1_wAngle1 { get; internal set; }
            public double dX2_wMagnitude2 { get; internal set; }
            public double dX2_wAngle2 { get; internal set; }
            public double dY2_wMagnitude2 { get; internal set; }
            public double dY2_wAngle2 { get; internal set; }
            public double dX3_wMagnitude3 { get; internal set; }
            public double dX3_wAngle3 { get; internal set; }
            public double dY3_wMagnitude3 { get; internal set; }
            public double dY3_wAngle3 { get; internal set; }
            public double dX4_wMagnitude4 { get; internal set; }
            public double dX4_wAngle4 { get; internal set; }
            public double dY4_wMagnitude4 { get; internal set; }
            public double dY4_wAngle4 { get; internal set; }
            public double dsumx_dX { get; internal set; }
            public double dsumx_dXPivot { get; internal set; }
            public double dsumx_dWeight { get; internal set; }
            public double dsumy_dY { get; internal set; }
            public double dsumy_dYPivot { get; internal set; }
            public double dsumy_dWeight { get; internal set; }
            public double dDiffX1_dSumX { get; internal set; }
            public double dDiffX1_dX1 { get; internal set; }
            public double dDiffY1_dSumY { get; internal set; }
            public double dDiffY1_dY1 { get; internal set; }
            public double dDiffX2_dSumX { get; internal set; }
            public double dDiffX2_dX2 { get; internal set; }
            public double dDiffY2_dSumY { get; internal set; }
            public double dDiffY2_dY2 { get; internal set; }
            public double dDiffX3_dSumX { get; internal set; }
            public double dDiffX3_dX3 { get; internal set; }
            public double dDiffY3_dSumY { get; internal set; }
            public double dDiffY3_dY3 { get; internal set; }
            public double dDiffX4_dSumX { get; internal set; }
            public double dDiffX4_dX4 { get; internal set; }
            public double dDiffY4_dSumY { get; internal set; }
            public double dDiffY4_dY4 { get; internal set; }
            public double dResultMagnitude1_dDiffX1 { get; internal set; }
            public double dResultMagnitude1_dDiffY1 { get; internal set; }
            public double dResultAngle1_dDiffX1 { get; internal set; }
            public double dResultAngle1_dDiffY1 { get; internal set; }
            public double dResultMagnitude2_dDiffX2 { get; internal set; }
            public double dResultMagnitude2_dDiffY2 { get; internal set; }
            public double dResultAngle2_dDiffX2 { get; internal set; }
            public double dResultAngle2_dDiffY2 { get; internal set; }
            public double dResultMagnitude3_dDiffX3 { get; internal set; }
            public double dResultMagnitude3_dDiffY3 { get; internal set; }
            public double dResultAngle3_dDiffX3 { get; internal set; }
            public double dResultAngle3_dDiffY3 { get; internal set; }
            public double dResultMagnitude4_dDiffX4 { get; internal set; }
            public double dResultMagnitude4_dDiffY4 { get; internal set; }
            public double dResultAngle4_dDiffX4 { get; internal set; }
            public double dResultAngle4_dDiffY4 { get; internal set; }
        }
    }
}
