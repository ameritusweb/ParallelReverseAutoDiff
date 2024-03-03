//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCartesianTiledSummationOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using ParallelReverseAutoDiff.GravNetExample.Common;
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise cartesian tiled summation operation.
    /// </summary>
    public class ElementwiseVectorCartesianTiledSummationOperation : Operation
    {
        private Matrix[,] input1;
        private Matrix[,] input2;
        private Matrix[,] weights;
        private double[,][] summationX;
        private double[,][] summationY;
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
            return new ElementwiseVectorCartesianTiledSummationOperation();
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
            Matrix[,] brokenInput1 = CommonMatrixUtils.BreakIntoSections(input1, 8);
            Matrix[,] brokenInput2 = CommonMatrixUtils.BreakIntoSections(input2, 8);
            Matrix[,] brokenWeights = CommonMatrixUtils.BreakIntoSectionsExactly(weights, 8);
            this.calculatedValues = new CalculatedValues[brokenInput1.GetLength(0), brokenInput1.GetLength(1)][,];
            this.summationX = new double[brokenInput1.GetLength(0), brokenInput1.GetLength(1)][];
            this.summationY = new double[brokenInput1.GetLength(0), brokenInput1.GetLength(1)][];

            Parallel.For(0, brokenInput1.GetLength(0), i =>
            {
                for (int j = 0; j < brokenInput1.GetLength(1); j++)
                {
                    this.InnerForward(i, j, brokenInput1[i, j], brokenInput2[i, j], brokenWeights[i, j]);
                }
            });

            this.Output = CommonMatrixUtils.PieceTogether(this.output);

            return this.Output;
        }

        private void InnerForward(int ii, int jj, Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1[ii, jj] = input1;
            this.input2[ii, jj] = input2;
            this.weights[ii, jj] = weights;

            this.calculatedValues[ii, jj] = new CalculatedValues[input1.Rows, input1.Cols / 2];

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
                    var calculatedValues = this.calculatedValues[ii, jj][i, j];
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

                    double dResultMagnitude_dsumx = (sumx * weights[i, j]) / Math.Sqrt(sumx * sumx + sumy * sumy);
                    double dResultMagnitude_dsumy = (sumy * weights[i, j]) / Math.Sqrt(sumx * sumx + sumy * sumy);
                    double dResultAngle_dsumx = -sumy / (sumx * sumx + sumy * sumy);
                    double dResultAngle_dsumy = sumx / (sumx * sumx + sumy * sumy);

                    double dResultMagnitude_dAngle = dResultMagnitude_dsumx * dsumx_dAngle + dResultMagnitude_dsumy * dsumy_dAngle;
                    double dResultMagnitude_dWAngle = dResultMagnitude_dsumx * dsumx_dWAngle + dResultMagnitude_dsumy * dsumy_dWAngle;
                    double dResultAngle_dAngle = dResultAngle_dsumx * dsumx_dAngle + dResultAngle_dsumy * dsumy_dAngle;
                    double dResultAngle_dWAngle = dResultAngle_dsumx * dsumx_dWAngle + dResultAngle_dsumy * dsumy_dWAngle;

                    double dResultMagnitude_dMagnitude = dResultMagnitude_dsumx * dsumx_dMagnitude + dResultMagnitude_dsumy * dsumy_dMagnitude;
                    double dResultMagnitude_dWMagnitude = dResultMagnitude_dsumx * dsumx_dWMagnitude + dResultMagnitude_dsumy * dsumy_dWMagnitude;
                    double dResultAngle_dMagnitude = dResultAngle_dsumx * dsumx_dMagnitude + dResultAngle_dsumy * dsumy_dMagnitude;
                    double dResultAngle_dWMagnitude = dResultAngle_dsumx * dsumx_dWMagnitude + dResultAngle_dsumy * dsumy_dWMagnitude;

                    resultVectors[(i * (input1.Cols / 2)) + j, 0] = resultMagnitude;
                    resultVectors[(i * (input1.Cols / 2)) + j, 1] = resultAngle;

                    double localSumX = resultMagnitude * Math.Cos(resultAngle);
                    double localSumY = resultMagnitude * Math.Sin(resultAngle);

                    double localSumXFull = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j] * Math.Cos(resultAngle);
                    double localSumYFull = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j] * Math.Sin(resultAngle);

                    double dLocalSumX_dWeight = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * Math.Cos(resultAngle);
                    double dLocalSumY_dWeight = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * Math.Sin(resultAngle);

                    calculatedValues.DLocalSumX_DWeight = dLocalSumX_dWeight;
                    calculatedValues.DLocalSumY_DWeight = dLocalSumY_dWeight;

                    double dLocalSumX_dResultMagnitude = Math.Cos(resultAngle);
                    double dLocalSumX_dResultAngle = -resultMagnitude * Math.Sin(resultAngle);

                    double dLocalSumX_dAngle = dLocalSumX_dResultMagnitude * dResultMagnitude_dAngle + dLocalSumX_dResultAngle * dResultAngle_dAngle;
                    double dLocalSumX_dWAngle = dLocalSumX_dResultMagnitude * dResultMagnitude_dWAngle + dLocalSumX_dResultAngle * dResultAngle_dWAngle;
                    double dLocalSumX_dMagnitude = dLocalSumX_dResultMagnitude * dResultMagnitude_dMagnitude + dLocalSumX_dResultAngle * dResultAngle_dMagnitude;
                    double dLocalSumX_dWMagnitude = dLocalSumX_dResultMagnitude * dResultMagnitude_dWMagnitude + dLocalSumX_dResultAngle * dResultAngle_dWMagnitude;

                    calculatedValues.DLocalSumX_DAngle = dLocalSumX_dAngle;
                    calculatedValues.DLocalSumX_DWAngle = dLocalSumX_dWAngle;
                    calculatedValues.DLocalSumX_DMagnitude = dLocalSumX_dMagnitude;
                    calculatedValues.DLocalSumX_DWMagnitude = dLocalSumX_dWMagnitude;

                    double dLocalSumY_dResultMagnitude = Math.Sin(resultAngle);
                    double dLocalSumY_dResultAngle = resultMagnitude * Math.Cos(resultAngle);

                    double dLocalSumY_dAngle = dLocalSumY_dResultMagnitude * dResultMagnitude_dAngle + dLocalSumY_dResultAngle * dResultAngle_dAngle;
                    double dLocalSumY_dWAngle = dLocalSumY_dResultMagnitude * dResultMagnitude_dWAngle + dLocalSumY_dResultAngle * dResultAngle_dWAngle;
                    double dLocalSumY_dMagnitude = dLocalSumY_dResultMagnitude * dResultMagnitude_dMagnitude + dLocalSumY_dResultAngle * dResultAngle_dMagnitude;
                    double dLocalSumY_dWMagnitude = dLocalSumY_dResultMagnitude * dResultMagnitude_dWMagnitude + dLocalSumY_dResultAngle * dResultAngle_dWMagnitude;

                    calculatedValues.DLocalSumY_DAngle = dLocalSumY_dAngle;
                    calculatedValues.DLocalSumY_DWAngle = dLocalSumY_dWAngle;
                    calculatedValues.DLocalSumY_DMagnitude = dLocalSumY_dMagnitude;
                    calculatedValues.DLocalSumY_DWMagnitude = dLocalSumY_dWMagnitude;

                    sumX += localSumX;
                    sumY += localSumY;
                }

                summationX[i] = sumX;
                summationY[i] = sumY;
            });

            this.summationX[ii, jj] = summationX;
            this.summationY[ii, jj] = summationY;

            this.output[ii, jj][0, 0] = this.summationX[ii, jj].Sum();
            this.output[ii, jj][0, 1] = this.summationY[ii, jj].Sum();
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
                .AddInputGradient(CommonMatrixUtils.PieceTogetherExactly(this.dWeights))
                .Build();
        }

        private void InnerBackward(int ii, int jj, Matrix input1, Matrix input2, Matrix weights, Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            double dSummationXOutput = dOutput[0, 0]; // Gradient of the loss function with respect to the output X
            double dSummationYOutput = dOutput[0, 1];     // Gradient of the loss function with respect to the output Y

            // Updating gradients with respect to resultMagnitude and resultAngle
            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    var values = this.calculatedValues[ii, jj][i, j];

                    // Update dWeights with direct contributions from summationX and summationY
                    dWeights[i, j] = dSummationXOutput * values.DLocalSumX_DWeight + dSummationYOutput * values.DLocalSumY_DWeight;

                    // Apply chain rule to propagate back to dInput1 and dInput2
                    dInput1[i, j] = dSummationXOutput * values.DLocalSumX_DMagnitude + dSummationYOutput * values.DLocalSumY_DMagnitude;
                    dInput1[i, j + (input1.Cols / 2)] = dSummationXOutput * values.DLocalSumX_DAngle + dSummationYOutput * values.DLocalSumY_DAngle;

                    dInput2[i, j] = dSummationXOutput * values.DLocalSumX_DWMagnitude + dSummationYOutput * values.DLocalSumY_DWMagnitude;
                    dInput2[i, j + (input2.Cols / 2)] = dSummationXOutput * values.DLocalSumX_DWAngle + dSummationYOutput * values.DLocalSumY_DWAngle;
                }
            });

            this.dInput1[ii, jj] = dInput1;
            this.dInput2[ii, jj] = dInput2;
            this.dWeights[ii, jj] = dWeights;
        }

        private struct CalculatedValues
        {
            public double DLocalSumX_DAngle;
            public double DLocalSumX_DWAngle;
            public double DLocalSumX_DMagnitude;
            public double DLocalSumX_DWMagnitude;

            public double DLocalSumY_DAngle;
            public double DLocalSumY_DWAngle;
            public double DLocalSumY_DMagnitude;
            public double DLocalSumY_DWMagnitude;

            public double DLocalSumX_DWeight;
            public double DLocalSumY_DWeight;
        }
    }
}
