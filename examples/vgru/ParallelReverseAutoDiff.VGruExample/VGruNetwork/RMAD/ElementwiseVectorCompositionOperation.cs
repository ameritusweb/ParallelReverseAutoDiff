//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCompositionOperation.cs" author="ameritusweb" date="4/8/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Reflection.Metadata;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector composition operation.
    /// </summary>
    public class ElementwiseVectorCompositionOperation : Operation
    {
        private Matrix input1;
        private Matrix weights;
        private CalculatedValues[,] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorCompositionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector composition function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector composition operation.</param>
        /// <param name="weights">The second input to the element-wise vector composition operation.</param>
        /// <returns>The output of the element-wise vector composition operation.</returns>
        public Matrix Forward(Matrix input1, Matrix weights)
        {
            this.input1 = input1;
            this.weights = weights;

            this.calculatedValues = new CalculatedValues[this.input1.Rows, this.input1.Cols / 2];

            this.Output = new Matrix(this.input1.Rows, this.input1.Cols / 10);
            Parallel.For(0, input1.Rows, i =>
            {
                int k = 0;
                for (int j = 0; j < input1.Cols / 2; j += 10)
                {
                    var calcedValues = this.calculatedValues[i, k++];

                    double magnitude1 = input1[i, j] * weights[i, j];
                    double angle1 = input1[i, j + (input1.Cols / 2)];

                    double magnitude2 = input1[i, j + 1] * weights[i, j + 1];
                    double angle2 = input1[i, j + 1 + (input1.Cols / 2)];

                    double magnitude3 = input1[i, j + 2] * weights[i, j + 2];
                    double angle3 = input1[i, j + 2 + (input1.Cols / 2)];

                    double magnitude4 = input1[i, j + 3] * weights[i, j + 3];
                    double angle4 = input1[i, j + 3 + (input1.Cols / 2)];

                    double magnitude5 = input1[i, j + 4] * weights[i, j + 4];
                    double angle5 = input1[i, j + 4 + (input1.Cols / 2)];

                    double magnitude6 = input1[i, j + 5] * weights[i, j + 5];
                    double angle6 = input1[i, j + 5 + (input1.Cols / 2)];

                    double magnitude7 = input1[i, j + 6] * weights[i, j + 6];
                    double angle7 = input1[i, j + 6 + (input1.Cols / 2)];

                    double magnitude8 = input1[i, j + 7] * weights[i, j + 7];
                    double angle8 = input1[i, j + 7 + (input1.Cols / 2)];

                    double magnitude9 = input1[i, j + 8] * weights[i, j + 8];
                    double angle9 = input1[i, j + 8 + (input1.Cols / 2)];

                    double magnitude10 = input1[i, j + 9] * weights[i, j + 9];
                    double angle10 = input1[i, j + 9 + (input1.Cols / 2)];

                    var x1 = magnitude1 * Math.Cos(angle1);
                    var y1 = magnitude1 * Math.Sin(angle1);

                    var x2 = magnitude2 * Math.Cos(angle2);
                    var y2 = magnitude2 * Math.Sin(angle2);

                    var x3 = magnitude3 * Math.Cos(angle3);
                    var y3 = magnitude3 * Math.Sin(angle3);

                    var x4 = magnitude4 * Math.Cos(angle4);
                    var y4 = magnitude4 * Math.Sin(angle4);

                    var x5 = magnitude5 * Math.Cos(angle5);
                    var y5 = magnitude5 * Math.Sin(angle5);

                    var x6 = magnitude6 * Math.Cos(angle6);
                    var y6 = magnitude6 * Math.Sin(angle6);

                    var x7 = magnitude7 * Math.Cos(angle7);
                    var y7 = magnitude7 * Math.Sin(angle7);

                    var x8 = magnitude8 * Math.Cos(angle8);
                    var y8 = magnitude8 * Math.Sin(angle8);

                    var x9 = magnitude9 * Math.Cos(angle9);
                    var y9 = magnitude9 * Math.Sin(angle9);

                    var x10 = magnitude10 * Math.Cos(angle10);
                    var y10 = magnitude10 * Math.Sin(angle10);

                    var x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10;
                    var y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10;

                    var dX_dWeight1 = input1[i, j] * Math.Cos(angle1);
                    var dY_dWeight1 = input1[i, j] * Math.Sin(angle1);
                    calcedValues.CV_dX_dWeight1 = dX_dWeight1;
                    calcedValues.CV_dY_dWeight1 = dY_dWeight1;

                    var dX_dWeight2 = input1[i, j + 1] * Math.Cos(angle2);
                    var dY_dWeight2 = input1[i, j + 1] * Math.Sin(angle2);
                    calcedValues.CV_dX_dWeight2 = dX_dWeight2;
                    calcedValues.CV_dY_dWeight2 = dY_dWeight2;

                    var dX_dWeight3 = input1[i, j + 2] * Math.Cos(angle3);
                    var dY_dWeight3 = input1[i, j + 2] * Math.Sin(angle3);
                    calcedValues.CV_dX_dWeight3 = dX_dWeight3;
                    calcedValues.CV_dY_dWeight3 = dY_dWeight3;

                    var dX_dWeight4 = input1[i, j + 3] * Math.Cos(angle4);
                    var dY_dWeight4 = input1[i, j + 3] * Math.Sin(angle4);
                    calcedValues.CV_dX_dWeight4 = dX_dWeight4;
                    calcedValues.CV_dY_dWeight4 = dY_dWeight4;

                    var dX_dWeight5 = input1[i, j + 4] * Math.Cos(angle5);
                    var dY_dWeight5 = input1[i, j + 4] * Math.Sin(angle5);
                    calcedValues.CV_dX_dWeight5 = dX_dWeight5;
                    calcedValues.CV_dY_dWeight5 = dY_dWeight5;

                    var dX_dWeight6 = input1[i, j + 5] * Math.Cos(angle6);
                    var dY_dWeight6 = input1[i, j + 5] * Math.Sin(angle6);
                    calcedValues.CV_dX_dWeight6 = dX_dWeight6;
                    calcedValues.CV_dY_dWeight6 = dY_dWeight6;

                    var dX_dWeight7 = input1[i, j + 6] * Math.Cos(angle7);
                    var dY_dWeight7 = input1[i, j + 6] * Math.Sin(angle7);
                    calcedValues.CV_dX_dWeight7 = dX_dWeight7;
                    calcedValues.CV_dY_dWeight7 = dY_dWeight7;

                    var dX_dWeight8 = input1[i, j + 7] * Math.Cos(angle8);
                    var dY_dWeight8 = input1[i, j + 7] * Math.Sin(angle8);
                    calcedValues.CV_dX_dWeight8 = dX_dWeight8;
                    calcedValues.CV_dY_dWeight8 = dY_dWeight8;

                    var dX_dWeight9 = input1[i, j + 8] * Math.Cos(angle9);
                    var dY_dWeight9 = input1[i, j + 8] * Math.Sin(angle9);
                    calcedValues.CV_dX_dWeight9 = dX_dWeight9;
                    calcedValues.CV_dY_dWeight9 = dY_dWeight9;

                    var dX_dWeight10 = input1[i, j + 9] * Math.Cos(angle10);
                    var dY_dWeight10 = input1[i, j + 9] * Math.Sin(angle10);
                    calcedValues.CV_dX_dWeight10 = dX_dWeight10;
                    calcedValues.CV_dY_dWeight10 = dY_dWeight10;

                    var dX_dInput1 = weights[i, j] * Math.Cos(angle1);
                    var dX_dInputAngle1 = -magnitude1 * Math.Sin(angle1);
                    calcedValues.CV_dx_dInput1 = dX_dInput1;
                    calcedValues.CV_dx_dInputAngle1 = dX_dInputAngle1;

                    var dY_dInput1 = weights[i, j] * Math.Sin(angle1);
                    var dY_dInputAngle1 = magnitude1 * Math.Cos(angle1);
                    calcedValues.CV_dy_dInput1 = dY_dInput1;
                    calcedValues.CV_dy_dInputAngle1 = dY_dInputAngle1;

                    var dX_dInput2 = weights[i, j + 1] * Math.Cos(angle2);
                    var dX_dInputAngle2 = -magnitude2 * Math.Sin(angle2);
                    calcedValues.CV_dx_dInput2 = dX_dInput2;
                    calcedValues.CV_dx_dInputAngle2 = dX_dInputAngle2;

                    var dY_dInput2 = weights[i, j + 1] * Math.Sin(angle2);
                    var dY_dInputAngle2 = magnitude2 * Math.Cos(angle2);
                    calcedValues.CV_dy_dInput2 = dY_dInput2;
                    calcedValues.CV_dy_dInputAngle2 = dY_dInputAngle2;

                    var dX_dInput3 = weights[i, j + 2] * Math.Cos(angle3);
                    var dX_dInputAngle3 = -magnitude3 * Math.Sin(angle3);
                    calcedValues.CV_dx_dInput3 = dX_dInput3;
                    calcedValues.CV_dx_dInputAngle3 = dX_dInputAngle3;

                    var dY_dInput3 = weights[i, j + 2] * Math.Sin(angle3);
                    var dY_dInputAngle3 = magnitude3 * Math.Cos(angle3);
                    calcedValues.CV_dy_dInput3 = dY_dInput3;
                    calcedValues.CV_dy_dInputAngle3 = dY_dInputAngle3;

                    var dX_dInput4 = weights[i, j + 3] * Math.Cos(angle4);
                    var dX_dInputAngle4 = -magnitude4 * Math.Sin(angle4);
                    calcedValues.CV_dx_dInput4 = dX_dInput4;
                    calcedValues.CV_dx_dInputAngle4 = dX_dInputAngle4;

                    var dY_dInput4 = weights[i, j + 3] * Math.Sin(angle4);
                    var dY_dInputAngle4 = magnitude4 * Math.Cos(angle4);
                    calcedValues.CV_dy_dInput4 = dY_dInput4;
                    calcedValues.CV_dy_dInputAngle4 = dY_dInputAngle4;

                    var dX_dInput5 = weights[i, j + 4] * Math.Cos(angle5);
                    var dX_dInputAngle5 = -magnitude5 * Math.Sin(angle5);
                    calcedValues.CV_dx_dInput5 = dX_dInput5;
                    calcedValues.CV_dx_dInputAngle5 = dX_dInputAngle5;

                    var dY_dInput5 = weights[i, j + 4] * Math.Sin(angle5);
                    var dY_dInputAngle5 = magnitude5 * Math.Cos(angle5);
                    calcedValues.CV_dy_dInput5 = dY_dInput5;
                    calcedValues.CV_dy_dInputAngle5 = dY_dInputAngle5;

                    var dX_dInput6 = weights[i, j + 5] * Math.Cos(angle6);
                    var dX_dInputAngle6 = -magnitude6 * Math.Sin(angle6);
                    calcedValues.CV_dx_dInput6 = dX_dInput6;
                    calcedValues.CV_dx_dInputAngle6 = dX_dInputAngle6;

                    var dY_dInput6 = weights[i, j + 5] * Math.Sin(angle6);
                    var dY_dInputAngle6 = magnitude6 * Math.Cos(angle6);
                    calcedValues.CV_dy_dInput6 = dY_dInput6;
                    calcedValues.CV_dy_dInputAngle6 = dY_dInputAngle6;

                    var dX_dInput7 = weights[i, j + 6] * Math.Cos(angle7);
                    var dX_dInputAngle7 = -magnitude7 * Math.Sin(angle7);
                    calcedValues.CV_dx_dInput7 = dX_dInput7;
                    calcedValues.CV_dx_dInputAngle7 = dX_dInputAngle7;

                    var dY_dInput7 = weights[i, j + 6] * Math.Sin(angle7);
                    var dY_dInputAngle7 = magnitude7 * Math.Cos(angle7);
                    calcedValues.CV_dy_dInput7 = dY_dInput7;
                    calcedValues.CV_dy_dInputAngle7 = dY_dInputAngle7;

                    var dX_dInput8 = weights[i, j + 7] * Math.Cos(angle8);
                    var dX_dInputAngle8 = -magnitude8 * Math.Sin(angle8);
                    calcedValues.CV_dx_dInput8 = dX_dInput8;
                    calcedValues.CV_dx_dInputAngle8 = dX_dInputAngle8;

                    var dY_dInput8 = weights[i, j + 7] * Math.Sin(angle8);
                    var dY_dInputAngle8 = magnitude8 * Math.Cos(angle8);
                    calcedValues.CV_dy_dInput8 = dY_dInput8;
                    calcedValues.CV_dy_dInputAngle8 = dY_dInputAngle8;

                    var dX_dInput9 = weights[i, j + 8] * Math.Cos(angle9);
                    var dX_dInputAngle9 = -magnitude9 * Math.Sin(angle9);
                    calcedValues.CV_dx_dInput9 = dX_dInput9;
                    calcedValues.CV_dx_dInputAngle9 = dX_dInputAngle9;

                    var dY_dInput9 = weights[i, j + 8] * Math.Sin(angle9);
                    var dY_dInputAngle9 = magnitude9 * Math.Cos(angle9);
                    calcedValues.CV_dy_dInput9 = dY_dInput9;
                    calcedValues.CV_dy_dInputAngle9 = dY_dInputAngle9;

                    var dX_dInput10 = weights[i, j + 9] * Math.Cos(angle10);
                    var dX_dInputAngle10 = -magnitude10 * Math.Sin(angle10);
                    calcedValues.CV_dx_dInput10 = dX_dInput10;
                    calcedValues.CV_dx_dInputAngle10 = dX_dInputAngle10;

                    var dY_dInput10 = weights[i, j + 9] * Math.Sin(angle10);
                    var dY_dInputAngle10 = magnitude10 * Math.Cos(angle10);
                    calcedValues.CV_dy_dInput10 = dY_dInput10;
                    calcedValues.CV_dy_dInputAngle10 = dY_dInputAngle10;

                    var resultMagnitude = Math.Sqrt((x * x) + (y * y));
                    var resultAngle = Math.Atan2(y, x);

                    var dResultMagnitude_dX = x / resultMagnitude;
                    var dResultMagnitude_dY = y / resultMagnitude;
                    var dResultAngle_dX = -y / ((x * x) + (y * y));
                    var dResultAngle_dY = x / ((x * x) + (y * y));
                    calcedValues.CV_dResultMagnitude_dX = dResultMagnitude_dX;
                    calcedValues.CV_dResultMagnitude_dY = dResultMagnitude_dY;
                    calcedValues.CV_dResultAngle_dX = dResultAngle_dX;
                    calcedValues.CV_dResultAngle_dY = dResultAngle_dY;

                    this.Output[i, j / 10] = resultMagnitude;
                    this.Output[i, (j / 10) + (this.input1.Cols / 20)] = resultAngle;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                int k = 0;
                for (int j = 0; j < this.input1.Cols / 2; j += 10)
                {
                    var calcedValues = this.calculatedValues[i, k++];

                    // Assuming dOutput is organized similarly to Output, with magnitudes and angles interlaced
                    double dOutputMagnitude = dOutput[i, j / 10];
                    double dOutputAngle = dOutput[i, (j / 10) + (this.input1.Cols / 20)];

                    double[] xValues = new double[10];
                    xValues[0] = calcedValues.CV_dx_dInput1;
                    xValues[1] = calcedValues.CV_dx_dInput2;
                    xValues[2] = calcedValues.CV_dx_dInput3;
                    xValues[3] = calcedValues.CV_dx_dInput4;
                    xValues[4] = calcedValues.CV_dx_dInput5;
                    xValues[5] = calcedValues.CV_dx_dInput6;
                    xValues[6] = calcedValues.CV_dx_dInput7;
                    xValues[7] = calcedValues.CV_dx_dInput8;
                    xValues[8] = calcedValues.CV_dx_dInput9;
                    xValues[9] = calcedValues.CV_dx_dInput10;

                    double[] yValues = new double[10];
                    yValues[0] = calcedValues.CV_dy_dInput1;
                    yValues[1] = calcedValues.CV_dy_dInput2;
                    yValues[2] = calcedValues.CV_dy_dInput3;
                    yValues[3] = calcedValues.CV_dy_dInput4;
                    yValues[4] = calcedValues.CV_dy_dInput5;
                    yValues[5] = calcedValues.CV_dy_dInput6;
                    yValues[6] = calcedValues.CV_dy_dInput7;
                    yValues[7] = calcedValues.CV_dy_dInput8;
                    yValues[8] = calcedValues.CV_dy_dInput9;
                    yValues[9] = calcedValues.CV_dy_dInput10;

                    double[] xAngleValues = new double[10];
                    xAngleValues[0] = calcedValues.CV_dx_dInputAngle1;
                    xAngleValues[1] = calcedValues.CV_dx_dInputAngle2;
                    xAngleValues[2] = calcedValues.CV_dx_dInputAngle3;
                    xAngleValues[3] = calcedValues.CV_dx_dInputAngle4;
                    xAngleValues[4] = calcedValues.CV_dx_dInputAngle5;
                    xAngleValues[5] = calcedValues.CV_dx_dInputAngle6;
                    xAngleValues[6] = calcedValues.CV_dx_dInputAngle7;
                    xAngleValues[7] = calcedValues.CV_dx_dInputAngle8;
                    xAngleValues[8] = calcedValues.CV_dx_dInputAngle9;
                    xAngleValues[9] = calcedValues.CV_dx_dInputAngle10;

                    double[] yAngleValues = new double[10];
                    yAngleValues[0] = calcedValues.CV_dy_dInputAngle1;
                    yAngleValues[1] = calcedValues.CV_dy_dInputAngle2;
                    yAngleValues[2] = calcedValues.CV_dy_dInputAngle3;
                    yAngleValues[3] = calcedValues.CV_dy_dInputAngle4;
                    yAngleValues[4] = calcedValues.CV_dy_dInputAngle5;
                    yAngleValues[5] = calcedValues.CV_dy_dInputAngle6;
                    yAngleValues[6] = calcedValues.CV_dy_dInputAngle7;
                    yAngleValues[7] = calcedValues.CV_dy_dInputAngle8;
                    yAngleValues[8] = calcedValues.CV_dy_dInputAngle9;
                    yAngleValues[9] = calcedValues.CV_dy_dInputAngle10;

                    double[] xWeightValues = new double[10];
                    xWeightValues[0] = calcedValues.CV_dX_dWeight1;
                    xWeightValues[1] = calcedValues.CV_dX_dWeight2;
                    xWeightValues[2] = calcedValues.CV_dX_dWeight3;
                    xWeightValues[3] = calcedValues.CV_dX_dWeight4;
                    xWeightValues[4] = calcedValues.CV_dX_dWeight5;
                    xWeightValues[5] = calcedValues.CV_dX_dWeight6;
                    xWeightValues[6] = calcedValues.CV_dX_dWeight7;
                    xWeightValues[7] = calcedValues.CV_dX_dWeight8;
                    xWeightValues[8] = calcedValues.CV_dX_dWeight9;
                    xWeightValues[9] = calcedValues.CV_dX_dWeight10;

                    double[] yWeightValues = new double[10];
                    yWeightValues[0] = calcedValues.CV_dY_dWeight1;
                    yWeightValues[1] = calcedValues.CV_dY_dWeight2;
                    yWeightValues[2] = calcedValues.CV_dY_dWeight3;
                    yWeightValues[3] = calcedValues.CV_dY_dWeight4;
                    yWeightValues[4] = calcedValues.CV_dY_dWeight5;
                    yWeightValues[5] = calcedValues.CV_dY_dWeight6;
                    yWeightValues[6] = calcedValues.CV_dY_dWeight7;
                    yWeightValues[7] = calcedValues.CV_dY_dWeight8;
                    yWeightValues[8] = calcedValues.CV_dY_dWeight9;
                    yWeightValues[9] = calcedValues.CV_dY_dWeight10;

                    // Update gradients for each vector component based on chain rule
                    for (int component = 0; component < 10; component++)
                    {
                        int magnitudeIndex = j + component;
                        int angleIndex = magnitudeIndex + (this.input1.Cols / 2);

                        // Accumulate the gradients for input magnitudes and angles
                        dInput1[i, magnitudeIndex] += dOutputMagnitude * calcedValues.CV_dResultMagnitude_dX * xValues[component];
                        dInput1[i, magnitudeIndex] += dOutputMagnitude * calcedValues.CV_dResultMagnitude_dY * yValues[component];
                        dInput1[i, magnitudeIndex] += dOutputAngle * calcedValues.CV_dResultAngle_dX * xValues[component];
                        dInput1[i, magnitudeIndex] += dOutputAngle * calcedValues.CV_dResultAngle_dY * yValues[component];

                        dInput1[i, angleIndex] += dOutputMagnitude * calcedValues.CV_dResultMagnitude_dX * xAngleValues[component];
                        dInput1[i, angleIndex] += dOutputMagnitude * calcedValues.CV_dResultMagnitude_dY * yAngleValues[component];
                        dInput1[i, angleIndex] += dOutputAngle * calcedValues.CV_dResultAngle_dX * xAngleValues[component];
                        dInput1[i, angleIndex] += dOutputAngle * calcedValues.CV_dResultAngle_dY * yAngleValues[component];

                        // Update gradients for weights
                        dWeights[i, magnitudeIndex] += dOutputMagnitude * calcedValues.CV_dResultMagnitude_dX * xWeightValues[component];
                        dWeights[i, magnitudeIndex] += dOutputMagnitude * calcedValues.CV_dResultMagnitude_dY * yWeightValues[component];
                        dWeights[i, magnitudeIndex] += dOutputAngle * calcedValues.CV_dResultAngle_dX * xWeightValues[component];
                        dWeights[i, magnitudeIndex] += dOutputAngle * calcedValues.CV_dResultAngle_dY * yWeightValues[component];
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dWeights)
                .Build();
        }

        private struct CalculatedValues
        {
            public double CV_dx_dInput1 { get; set; }

            public double CV_dx_dInputAngle1 { get; set; }

            public double CV_dy_dInput1 { get; set; }

            public double CV_dy_dInputAngle1 { get; set; }

            public double CV_dx_dInput2 { get; set; }

            public double CV_dx_dInputAngle2 { get; set; }

            public double CV_dy_dInput2 { get; set; }

            public double CV_dy_dInputAngle2 { get; set; }

            public double CV_dx_dInput3 { get; set; }

            public double CV_dx_dInputAngle3 { get; set; }

            public double CV_dy_dInput3 { get; set; }

            public double CV_dy_dInputAngle3 { get; set; }

            public double CV_dx_dInput4 { get; set; }

            public double CV_dx_dInputAngle4 { get; set; }

            public double CV_dy_dInput4 { get; set; }

            public double CV_dy_dInputAngle4 { get; set; }

            public double CV_dx_dInput5 { get; set; }

            public double CV_dx_dInputAngle5 { get; set; }

            public double CV_dy_dInput5 { get; set; }

            public double CV_dy_dInputAngle5 { get; set; }

            public double CV_dx_dInput6 { get; set; }

            public double CV_dx_dInputAngle6 { get; set; }

            public double CV_dy_dInput6 { get; set; }

            public double CV_dy_dInputAngle6 { get; set; }

            public double CV_dx_dInput7 { get; set; }

            public double CV_dx_dInputAngle7 { get; set; }

            public double CV_dy_dInput7 { get; set; }

            public double CV_dy_dInputAngle7 { get; set; }

            public double CV_dx_dInput8 { get; set; }

            public double CV_dx_dInputAngle8 { get; set; }

            public double CV_dy_dInput8 { get; set; }

            public double CV_dy_dInputAngle8 { get; set; }

            public double CV_dx_dInput9 { get; set; }

            public double CV_dx_dInputAngle9 { get; set; }

            public double CV_dy_dInput9 { get; set; }

            public double CV_dy_dInputAngle9 { get; set; }

            public double CV_dx_dInput10 { get; set; }

            public double CV_dx_dInputAngle10 { get; set; }

            public double CV_dy_dInput10 { get; set; }

            public double CV_dy_dInputAngle10 { get; set; }

            public double CV_dResultMagnitude_dX { get; set; }

            public double CV_dResultMagnitude_dY { get; set; }

            public double CV_dResultAngle_dX { get; set; }

            public double CV_dResultAngle_dY { get; set; }

            public double CV_dX_dWeight1 { get; set; }

            public double CV_dY_dWeight1 { get; set; }

            public double CV_dX_dWeight2 { get; set; }

            public double CV_dY_dWeight2 { get; set; }

            public double CV_dX_dWeight3 { get; set; }

            public double CV_dY_dWeight3 { get; set; }

            public double CV_dX_dWeight4 { get; set; }

            public double CV_dY_dWeight4 { get; set; }

            public double CV_dX_dWeight5 { get; set; }

            public double CV_dY_dWeight5 { get; set; }

            public double CV_dX_dWeight6 { get; set; }

            public double CV_dY_dWeight6 { get; set; }

            public double CV_dX_dWeight7 { get; set; }

            public double CV_dY_dWeight7 { get; set; }

            public double CV_dX_dWeight8 { get; set; }

            public double CV_dY_dWeight8 { get; set; }

            public double CV_dX_dWeight9 { get; set; }

            public double CV_dY_dWeight9 { get; set; }

            public double CV_dX_dWeight10 { get; set; }

            public double CV_dY_dWeight10 { get; set; }
        }
    }
}
