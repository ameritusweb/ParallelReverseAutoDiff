//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCartesianSummationOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise cartesian summation operation.
    /// </summary>
    public class ElementwiseVectorCartesianSummationOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private double[] summationX;
        private double[] summationY;
        private CalculatedValues[,] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorCartesianSummationOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector summation function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector summation operation.</param>
        /// <param name="input2">The second input to the element-wise vector summation operation.</param>
        /// <param name="weights">The weights.</param>
        /// <returns>The output of the element-wise vector summation operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(1, 2);

            this.calculatedValues = new CalculatedValues[this.input1.Rows, this.input1.Cols / 2];

            double[] summationX = new double[input1.Rows];
            double[] summationY = new double[input1.Rows];
            double[,] resultVectors = new double[input1.Rows * (input1.Cols / 2), 2];
            Parallel.For(0, input1.Rows, i =>
            {
                double sumX = 0.0d;
                double sumY = 0.0d;
                (double, double)[] resultMagnitudes = new (double, double)[input1.Cols / 2];
                for (int j = 0; j < (input1.Cols / 2); j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];

                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + (input2.Cols / 2)];

                    // Compute vector components
                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    double dsumx_dAngle = -magnitude * Math.Sin(angle);
                    double dsumx_dWAngle = -wMagnitude * Math.Sin(wAngle);
                    double dsumy_dAngle = magnitude * Math.Cos(angle);
                    double dsumy_dWAngle = wMagnitude * Math.Cos(wAngle);
                    double dsumx_dMagnitude = Math.Cos(angle);
                    double dsumx_dWMagnitude = Math.Cos(wAngle);
                    double dsumy_dMagnitude = Math.Sin(angle);
                    double dsumy_dWMagnitude = Math.Sin(wAngle);

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    double resultAngle = Math.Atan2(sumy, sumx);

                    double dResultMagnitude_dsumx = (sumx * weights[i, j]) / Math.Sqrt((sumx * sumx) + (sumy * sumy));
                    double dResultMagnitude_dsumy = (sumy * weights[i, j]) / Math.Sqrt((sumx * sumx) + (sumy * sumy));
                    double dResultAngle_dsumx = -sumy / ((sumx * sumx) + (sumy * sumy));
                    double dResultAngle_dsumy = sumx / ((sumx * sumx) + (sumy * sumy));

                    double dResultMagnitude_dAngle = (dResultMagnitude_dsumx * dsumx_dAngle) + (dResultMagnitude_dsumy * dsumy_dAngle);
                    double dResultMagnitude_dWAngle = (dResultMagnitude_dsumx * dsumx_dWAngle) + (dResultMagnitude_dsumy * dsumy_dWAngle);
                    double dResultAngle_dAngle = (dResultAngle_dsumx * dsumx_dAngle) + (dResultAngle_dsumy * dsumy_dAngle);
                    double dResultAngle_dWAngle = (dResultAngle_dsumx * dsumx_dWAngle) + (dResultAngle_dsumy * dsumy_dWAngle);

                    double dResultMagnitude_dMagnitude = (dResultMagnitude_dsumx * dsumx_dMagnitude) + (dResultMagnitude_dsumy * dsumy_dMagnitude);
                    double dResultMagnitude_dWMagnitude = (dResultMagnitude_dsumx * dsumx_dWMagnitude) + (dResultMagnitude_dsumy * dsumy_dWMagnitude);
                    double dResultAngle_dMagnitude = (dResultAngle_dsumx * dsumx_dMagnitude) + (dResultAngle_dsumy * dsumy_dMagnitude);
                    double dResultAngle_dWMagnitude = (dResultAngle_dsumx * dsumx_dWMagnitude) + (dResultAngle_dsumy * dsumy_dWMagnitude);

                    resultVectors[(i * (input1.Cols / 2)) + j, 0] = resultMagnitude;
                    resultVectors[(i * (input1.Cols / 2)) + j, 1] = resultAngle;

                    double localSumX = resultMagnitude * Math.Cos(resultAngle);
                    double localSumY = resultMagnitude * Math.Sin(resultAngle);

                    double localSumXFull = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j] * Math.Cos(resultAngle);
                    double localSumYFull = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j] * Math.Sin(resultAngle);

                    double dLocalSumX_dWeight = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * Math.Cos(resultAngle);
                    double dLocalSumY_dWeight = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * Math.Sin(resultAngle);

                    this.calculatedValues[i, j].DLocalSumX_DWeight = dLocalSumX_dWeight;
                    this.calculatedValues[i, j].DLocalSumY_DWeight = dLocalSumY_dWeight;

                    double dLocalSumX_dResultMagnitude = Math.Cos(resultAngle);
                    double dLocalSumX_dResultAngle = -resultMagnitude * Math.Sin(resultAngle);

                    double dLocalSumX_dAngle = (dLocalSumX_dResultMagnitude * dResultMagnitude_dAngle) + (dLocalSumX_dResultAngle * dResultAngle_dAngle);
                    double dLocalSumX_dWAngle = (dLocalSumX_dResultMagnitude * dResultMagnitude_dWAngle) + (dLocalSumX_dResultAngle * dResultAngle_dWAngle);
                    double dLocalSumX_dMagnitude = (dLocalSumX_dResultMagnitude * dResultMagnitude_dMagnitude) + (dLocalSumX_dResultAngle * dResultAngle_dMagnitude);
                    double dLocalSumX_dWMagnitude = (dLocalSumX_dResultMagnitude * dResultMagnitude_dWMagnitude) + (dLocalSumX_dResultAngle * dResultAngle_dWMagnitude);

                    this.calculatedValues[i, j].DLocalSumX_DAngle = dLocalSumX_dAngle;
                    this.calculatedValues[i, j].DLocalSumX_DWAngle = dLocalSumX_dWAngle;
                    this.calculatedValues[i, j].DLocalSumX_DMagnitude = dLocalSumX_dMagnitude;
                    this.calculatedValues[i, j].DLocalSumX_DWMagnitude = dLocalSumX_dWMagnitude;

                    double dLocalSumY_dResultMagnitude = Math.Sin(resultAngle);
                    double dLocalSumY_dResultAngle = resultMagnitude * Math.Cos(resultAngle);

                    double dLocalSumY_dAngle = (dLocalSumY_dResultMagnitude * dResultMagnitude_dAngle) + (dLocalSumY_dResultAngle * dResultAngle_dAngle);
                    double dLocalSumY_dWAngle = (dLocalSumY_dResultMagnitude * dResultMagnitude_dWAngle) + (dLocalSumY_dResultAngle * dResultAngle_dWAngle);
                    double dLocalSumY_dMagnitude = (dLocalSumY_dResultMagnitude * dResultMagnitude_dMagnitude) + (dLocalSumY_dResultAngle * dResultAngle_dMagnitude);
                    double dLocalSumY_dWMagnitude = (dLocalSumY_dResultMagnitude * dResultMagnitude_dWMagnitude) + (dLocalSumY_dResultAngle * dResultAngle_dWMagnitude);

                    this.calculatedValues[i, j].DLocalSumY_DAngle = dLocalSumY_dAngle;
                    this.calculatedValues[i, j].DLocalSumY_DWAngle = dLocalSumY_dWAngle;
                    this.calculatedValues[i, j].DLocalSumY_DMagnitude = dLocalSumY_dMagnitude;
                    this.calculatedValues[i, j].DLocalSumY_DWMagnitude = dLocalSumY_dWMagnitude;

                    sumX += localSumX;
                    sumY += localSumY;
                }

                summationX[i] = sumX;
                summationY[i] = sumY;
            });

            this.summationX = summationX;
            this.summationY = summationY;

            this.Output[0, 0] = this.summationX.Sum();
            this.Output[0, 1] = this.summationY.Sum();

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            double dSummationXOutput = dOutput[0, 0]; // Gradient of the loss function with respect to the output X
            double dSummationYOutput = dOutput[0, 1];     // Gradient of the loss function with respect to the output Y

            // Updating gradients with respect to resultMagnitude and resultAngle
            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    var values = this.calculatedValues[i, j];

                    // Update dWeights with direct contributions from summationX and summationY
                    dWeights[i, j] = (dSummationXOutput * values.DLocalSumX_DWeight) + (dSummationYOutput * values.DLocalSumY_DWeight);

                    // Apply chain rule to propagate back to dInput1 and dInput2
                    dInput1[i, j] = (dSummationXOutput * values.DLocalSumX_DMagnitude) + (dSummationYOutput * values.DLocalSumY_DMagnitude);
                    dInput1[i, j + (this.input1.Cols / 2)] = (dSummationXOutput * values.DLocalSumX_DAngle) + (dSummationYOutput * values.DLocalSumY_DAngle);

                    dInput2[i, j] = (dSummationXOutput * values.DLocalSumX_DWMagnitude) + (dSummationYOutput * values.DLocalSumY_DWMagnitude);
                    dInput2[i, j + (this.input2.Cols / 2)] = (dSummationXOutput * values.DLocalSumX_DWAngle) + (dSummationYOutput * values.DLocalSumY_DWAngle);
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
            public double DLocalSumX_DAngle { get; internal set; }

            public double DLocalSumX_DWAngle { get; internal set; }

            public double DLocalSumX_DMagnitude { get; internal set; }

            public double DLocalSumX_DWMagnitude { get; internal set; }

            public double DLocalSumY_DAngle { get; internal set; }

            public double DLocalSumY_DWAngle { get; internal set; }

            public double DLocalSumY_DMagnitude { get; internal set; }

            public double DLocalSumY_DWMagnitude { get; internal set; }

            public double DLocalSumX_DWeight { get; internal set; }

            public double DLocalSumY_DWeight { get; internal set; }
        }
    }
}
