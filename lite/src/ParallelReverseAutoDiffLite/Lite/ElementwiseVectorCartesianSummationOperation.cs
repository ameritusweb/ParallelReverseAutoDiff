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
        private float[] summationX;
        private float[] summationY;
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

            float[] summationX = new float[input1.Rows];
            float[] summationY = new float[input1.Rows];
            float[,] resultVectors = new float[input1.Rows * (input1.Cols / 2), 2];
            Parallel.For(0, input1.Rows, i =>
            {
                float sumX = 0.0f;
                float sumY = 0.0f;
                (float, float)[] resultMagnitudes = new (float, float)[input1.Cols / 2];
                for (int j = 0; j < (input1.Cols / 2); j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    float magnitude = input1[i, j];
                    float angle = input1[i, j + (input1.Cols / 2)];

                    float wMagnitude = input2[i, j];
                    float wAngle = input2[i, j + (input2.Cols / 2)];

                    // Compute vector components
                    float x1 = magnitude * PradMath.Cos(angle);
                    float y1 = magnitude * PradMath.Sin(angle);
                    float x2 = wMagnitude * PradMath.Cos(wAngle);
                    float y2 = wMagnitude * PradMath.Sin(wAngle);

                    float sumx = x1 + x2;
                    float sumy = y1 + y2;

                    float dsumx_dAngle = -magnitude * PradMath.Sin(angle);
                    float dsumx_dWAngle = -wMagnitude * PradMath.Sin(wAngle);
                    float dsumy_dAngle = magnitude * PradMath.Cos(angle);
                    float dsumy_dWAngle = wMagnitude * PradMath.Cos(wAngle);
                    float dsumx_dMagnitude = PradMath.Cos(angle);
                    float dsumx_dWMagnitude = PradMath.Cos(wAngle);
                    float dsumy_dMagnitude = PradMath.Sin(angle);
                    float dsumy_dWMagnitude = PradMath.Sin(wAngle);

                    // Compute resultant vector magnitude and angle
                    float resultMagnitude = PradMath.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    float resultAngle = PradMath.Atan2(sumy, sumx);

                    float dResultMagnitude_dsumx = (sumx * weights[i, j]) / PradMath.Sqrt((sumx * sumx) + (sumy * sumy));
                    float dResultMagnitude_dsumy = (sumy * weights[i, j]) / PradMath.Sqrt((sumx * sumx) + (sumy * sumy));
                    float dResultAngle_dsumx = -sumy / ((sumx * sumx) + (sumy * sumy));
                    float dResultAngle_dsumy = sumx / ((sumx * sumx) + (sumy * sumy));

                    float dResultMagnitude_dAngle = (dResultMagnitude_dsumx * dsumx_dAngle) + (dResultMagnitude_dsumy * dsumy_dAngle);
                    float dResultMagnitude_dWAngle = (dResultMagnitude_dsumx * dsumx_dWAngle) + (dResultMagnitude_dsumy * dsumy_dWAngle);
                    float dResultAngle_dAngle = (dResultAngle_dsumx * dsumx_dAngle) + (dResultAngle_dsumy * dsumy_dAngle);
                    float dResultAngle_dWAngle = (dResultAngle_dsumx * dsumx_dWAngle) + (dResultAngle_dsumy * dsumy_dWAngle);

                    float dResultMagnitude_dMagnitude = (dResultMagnitude_dsumx * dsumx_dMagnitude) + (dResultMagnitude_dsumy * dsumy_dMagnitude);
                    float dResultMagnitude_dWMagnitude = (dResultMagnitude_dsumx * dsumx_dWMagnitude) + (dResultMagnitude_dsumy * dsumy_dWMagnitude);
                    float dResultAngle_dMagnitude = (dResultAngle_dsumx * dsumx_dMagnitude) + (dResultAngle_dsumy * dsumy_dMagnitude);
                    float dResultAngle_dWMagnitude = (dResultAngle_dsumx * dsumx_dWMagnitude) + (dResultAngle_dsumy * dsumy_dWMagnitude);

                    resultVectors[(i * (input1.Cols / 2)) + j, 0] = resultMagnitude;
                    resultVectors[(i * (input1.Cols / 2)) + j, 1] = resultAngle;

                    float localSumX = resultMagnitude * PradMath.Cos(resultAngle);
                    float localSumY = resultMagnitude * PradMath.Sin(resultAngle);

                    float localSumXFull = PradMath.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j] * PradMath.Cos(resultAngle);
                    float localSumYFull = PradMath.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j] * PradMath.Sin(resultAngle);

                    float dLocalSumX_dWeight = PradMath.Sqrt((sumx * sumx) + (sumy * sumy)) * PradMath.Cos(resultAngle);
                    float dLocalSumY_dWeight = PradMath.Sqrt((sumx * sumx) + (sumy * sumy)) * PradMath.Sin(resultAngle);

                    this.calculatedValues[i, j].DLocalSumX_DWeight = dLocalSumX_dWeight;
                    this.calculatedValues[i, j].DLocalSumY_DWeight = dLocalSumY_dWeight;

                    float dLocalSumX_dResultMagnitude = PradMath.Cos(resultAngle);
                    float dLocalSumX_dResultAngle = -resultMagnitude * PradMath.Sin(resultAngle);

                    float dLocalSumX_dAngle = (dLocalSumX_dResultMagnitude * dResultMagnitude_dAngle) + (dLocalSumX_dResultAngle * dResultAngle_dAngle);
                    float dLocalSumX_dWAngle = (dLocalSumX_dResultMagnitude * dResultMagnitude_dWAngle) + (dLocalSumX_dResultAngle * dResultAngle_dWAngle);
                    float dLocalSumX_dMagnitude = (dLocalSumX_dResultMagnitude * dResultMagnitude_dMagnitude) + (dLocalSumX_dResultAngle * dResultAngle_dMagnitude);
                    float dLocalSumX_dWMagnitude = (dLocalSumX_dResultMagnitude * dResultMagnitude_dWMagnitude) + (dLocalSumX_dResultAngle * dResultAngle_dWMagnitude);

                    this.calculatedValues[i, j].DLocalSumX_DAngle = dLocalSumX_dAngle;
                    this.calculatedValues[i, j].DLocalSumX_DWAngle = dLocalSumX_dWAngle;
                    this.calculatedValues[i, j].DLocalSumX_DMagnitude = dLocalSumX_dMagnitude;
                    this.calculatedValues[i, j].DLocalSumX_DWMagnitude = dLocalSumX_dWMagnitude;

                    float dLocalSumY_dResultMagnitude = PradMath.Sin(resultAngle);
                    float dLocalSumY_dResultAngle = resultMagnitude * PradMath.Cos(resultAngle);

                    float dLocalSumY_dAngle = (dLocalSumY_dResultMagnitude * dResultMagnitude_dAngle) + (dLocalSumY_dResultAngle * dResultAngle_dAngle);
                    float dLocalSumY_dWAngle = (dLocalSumY_dResultMagnitude * dResultMagnitude_dWAngle) + (dLocalSumY_dResultAngle * dResultAngle_dWAngle);
                    float dLocalSumY_dMagnitude = (dLocalSumY_dResultMagnitude * dResultMagnitude_dMagnitude) + (dLocalSumY_dResultAngle * dResultAngle_dMagnitude);
                    float dLocalSumY_dWMagnitude = (dLocalSumY_dResultMagnitude * dResultMagnitude_dWMagnitude) + (dLocalSumY_dResultAngle * dResultAngle_dWMagnitude);

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

            float dSummationXOutput = dOutput[0, 0]; // Gradient of the loss function with respect to the output X
            float dSummationYOutput = dOutput[0, 1];     // Gradient of the loss function with respect to the output Y

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
            public float DLocalSumX_DAngle { get; internal set; }

            public float DLocalSumX_DWAngle { get; internal set; }

            public float DLocalSumX_DMagnitude { get; internal set; }

            public float DLocalSumX_DWMagnitude { get; internal set; }

            public float DLocalSumY_DAngle { get; internal set; }

            public float DLocalSumY_DWAngle { get; internal set; }

            public float DLocalSumY_DMagnitude { get; internal set; }

            public float DLocalSumY_DWMagnitude { get; internal set; }

            public float DLocalSumX_DWeight { get; internal set; }

            public float DLocalSumY_DWeight { get; internal set; }
        }
    }
}
