//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCartesianRowSummationOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise add operation.
    /// </summary>
    public class ElementwiseVectorCartesianRowSummationOperation : Operation
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
            return new ElementwiseVectorCartesianRowSummationOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector cartestian row summation function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector cartestian row summation operation.</param>
        /// <param name="input2">The second input to the element-wise vector cartestian row summation operation.</param>
        /// <param name="weights">The weights for the element-wise vector cartestian row summation operation.</param>
        /// <returns>The output of the element-wise vector cartestian row summation operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(this.input1.Rows, 2);

            this.calculatedValues = new CalculatedValues[this.input1.Rows, this.input1.Cols / 2];

            Parallel.For(0, input1.Rows, i =>
            {
                double sumX = 0.0d;
                double sumY = 0.0d;
                for (int j = 0; j < input1.Cols / 2; j++)
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

                    double dsumx_dAngle = -Math.Sin(angle);
                    double dsumy_dAngle = Math.Cos(angle);
                    double dsumx_dWAngle = -Math.Sin(wAngle);
                    double dsumy_dWAngle = Math.Cos(wAngle);
                    this.calculatedValues[i, j].Dsumx_DAngle = dsumx_dAngle;
                    this.calculatedValues[i, j].Dsumy_DAngle = dsumy_dAngle;
                    this.calculatedValues[i, j].Dsumx_DWAngle = dsumx_dWAngle;
                    this.calculatedValues[i, j].Dsumy_DWAngle = dsumy_dWAngle;

                    double dsumx_dMagnitude = Math.Cos(angle);
                    double dsumy_dMagnitude = Math.Sin(angle);
                    double dsumx_dWMagnitude = Math.Cos(wAngle);
                    double dsumy_dWMagnitude = Math.Sin(wAngle);
                    this.calculatedValues[i, j].Dsumx_DMagnitude = dsumx_dMagnitude;
                    this.calculatedValues[i, j].Dsumy_DMagnitude = dsumy_dMagnitude;
                    this.calculatedValues[i, j].Dsumx_DWMagnitude = dsumx_dWMagnitude;
                    this.calculatedValues[i, j].Dsumy_DWMagnitude = dsumy_dWMagnitude;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * this.weights[i, j];
                    double resultAngle = Math.Atan2(sumy, sumx);

                    double dResultMagnitude_dWeight = Math.Sqrt((sumx * sumx) + (sumy * sumy));
                    this.calculatedValues[i, j].DResultMagnitude_DWeight = dResultMagnitude_dWeight;

                    double dResultMagnitude_dsumx = this.weights[i, j] * sumx / resultMagnitude;
                    double dResultMagnitude_dsumy = this.weights[i, j] * sumy / resultMagnitude;
                    double dResultAngle_dsumx = -sumy / (sumx * sumx + sumy * sumy);
                    double dResultAngle_dsumy = sumx / (sumx * sumx + sumy * sumy);
                    this.calculatedValues[i, j].DResultMagnitude_Dsumx = dResultMagnitude_dsumx;
                    this.calculatedValues[i, j].DResultMagnitude_Dsumy = dResultMagnitude_dsumy;
                    this.calculatedValues[i, j].DResultAngle_Dsumx = dResultAngle_dsumx;
                    this.calculatedValues[i, j].DResultAngle_Dsumy = dResultAngle_dsumy;

                    double localSumX = resultMagnitude * Math.Cos(resultAngle);
                    double localSumY = resultMagnitude * Math.Sin(resultAngle);

                    double dLocalSumX_dResultMagnitude = Math.Cos(resultAngle);
                    double dLocalSumX_dResultAngle = -resultMagnitude * Math.Sin(resultAngle);
                    double dLocalSumY_dResultMagnitude = Math.Sin(resultAngle);
                    double dLocalSumY_dResultAngle = resultMagnitude * Math.Cos(resultAngle);
                    this.calculatedValues[i, j].DLocalSumX_DResultMagnitude = dLocalSumX_dResultMagnitude;
                    this.calculatedValues[i, j].DLocalSumX_DResultAngle = dLocalSumX_dResultAngle;
                    this.calculatedValues[i, j].DLocalSumY_DResultMagnitude = dLocalSumY_dResultMagnitude;
                    this.calculatedValues[i, j].DLocalSumY_DResultAngle = dLocalSumY_dResultAngle;

                    sumX += localSumX;
                    sumY += localSumY;
                }

                this.Output[i, 0] = sumX;
                this.Output[i, 1] = sumY;
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeight = new Matrix(this.weights.Rows, this.weights.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    var calcVals = this.calculatedValues[i, j];

                    // Compute the gradient for input1 magnitude
                    dInput1[i, j] += dOutput[i, 0] * calcVals.DLocalSumX_DResultMagnitude * calcVals.DResultMagnitude_Dsumx * calcVals.Dsumx_DMagnitude
                                   + dOutput[i, 1] * calcVals.DLocalSumY_DResultMagnitude * calcVals.DResultMagnitude_Dsumy * calcVals.Dsumy_DMagnitude;

                    // Compute the gradient for input1 angle
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, 0] * (calcVals.DLocalSumX_DResultAngle * calcVals.DResultAngle_Dsumx * calcVals.Dsumx_DAngle +
                                                                              calcVals.DLocalSumX_DResultMagnitude * calcVals.DResultMagnitude_Dsumx * calcVals.Dsumx_DAngle)
                                                               + dOutput[i, 1] * (calcVals.DLocalSumY_DResultAngle * calcVals.DResultAngle_Dsumy * calcVals.Dsumy_DAngle +
                                                                                  calcVals.DLocalSumY_DResultMagnitude * calcVals.DResultMagnitude_Dsumy * calcVals.Dsumy_DAngle);

                    // Compute the gradient for input2 magnitude
                    dInput2[i, j] += dOutput[i, 0] * calcVals.DLocalSumX_DResultMagnitude * calcVals.DResultMagnitude_Dsumx * calcVals.Dsumx_DWMagnitude
                                   + dOutput[i, 1] * calcVals.DLocalSumY_DResultMagnitude * calcVals.DResultMagnitude_Dsumy * calcVals.Dsumy_DWMagnitude;

                    // Compute the gradient for input2 angle
                    dInput2[i, j + (this.input2.Cols / 2)] += dOutput[i, 0] * (calcVals.DLocalSumX_DResultAngle * calcVals.DResultAngle_Dsumx * calcVals.Dsumx_DWAngle +
                                                                               calcVals.DLocalSumX_DResultMagnitude * calcVals.DResultMagnitude_Dsumx * calcVals.Dsumx_DWAngle)
                                                            + dOutput[i, 1] * (calcVals.DLocalSumY_DResultAngle * calcVals.DResultAngle_Dsumy * calcVals.Dsumy_DWAngle +
                                                                               calcVals.DLocalSumY_DResultMagnitude * calcVals.DResultMagnitude_Dsumy * calcVals.Dsumy_DWAngle);

                    // Gradient for weights
                    dWeight[i, j] += dOutput[i, 0] * calcVals.DLocalSumX_DResultMagnitude * calcVals.DResultMagnitude_DWeight
                                   + dOutput[i, 1] * calcVals.DLocalSumY_DResultMagnitude * calcVals.DResultMagnitude_DWeight;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeight)
                .Build();
        }

        private struct CalculatedValues
        {
            public double Dsumx_DMagnitude { get; set; }

            public double Dsumy_DMagnitude { get; set; }

            public double Dsumx_DWMagnitude { get; set; }

            public double Dsumy_DWMagnitude { get; set; }

            public double Dsumx_DAngle { get; set; }

            public double Dsumy_DAngle { get; set; }

            public double Dsumx_DWAngle { get; set; }

            public double Dsumy_DWAngle { get; set; }

            public double DResultMagnitude_DWeight { get; set; }

            public double DResultMagnitude_Dsumx { get; set; }

            public double DResultMagnitude_Dsumy { get; set; }

            public double DResultAngle_Dsumx { get; set; }

            public double DResultAngle_Dsumy { get; set; }

            public double DLocalSumX_DResultMagnitude { get; set; }

            public double DLocalSumX_DResultAngle { get; set; }

            public double DLocalSumY_DResultMagnitude { get; set; }

            public double DLocalSumY_DResultAngle { get; set; }
        }
    }
}
