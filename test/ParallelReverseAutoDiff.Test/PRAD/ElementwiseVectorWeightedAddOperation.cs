using ParallelReverseAutoDiff.RMAD;
using System.Diagnostics;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    /// <summary>
    /// Element-wise vector projection operation.
    /// </summary>
    public class ElementwiseVectorWeightedAddOperation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;

        /// <summary>
        /// Performs the forward operation for the element-wise vector summation function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector summation operation.</param>
        /// <param name="input2">The second input to the element-wise vector summation operation.</param>
        /// <param name="weights">The weights for the element-wise vector summation operation.</param>
        /// <returns>The output of the element-wise vector summation operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            var output = new Matrix(this.input1.Rows, this.input1.Cols);
            Parallel.For(0, input1.Rows, i =>
            {
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

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    double resultAngle = Math.Atan2(sumy, sumx);

                    output[i, j] = resultMagnitude;
                    output[i, j + (this.input1.Cols / 2)] = resultAngle;
                }
            });

            return output;
        }

        public BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    var magnitude = this.input1[i, j];  // magnitude of angle1
                    var angle = this.input1[i, j + (this.input1.Cols / 2)];
                    var wMagnitude = this.input2[i, j];  // magnitude of wAngle
                    var wAngle = this.input2[i, j + (this.input2.Cols / 2)];

                    var cAngle = Math.Cos(angle);
                    var sAngle = Math.Sin(angle);
                    var cwAngle = Math.Cos(wAngle);
                    var swAngle = Math.Sin(wAngle);

                    var x1 = magnitude * Math.Cos(angle);
                    var y1 = magnitude * Math.Sin(angle);
                    var x2 = wMagnitude * Math.Cos(wAngle);
                    var y2 = wMagnitude * Math.Sin(wAngle);

                    var combinedX = x1 + x2;
                    var combinedY = y1 + y2;

                    // Compute derivatives of combinedX and combinedY w.r.t. magnitudes and angles
                    var dCombinedX_dMagnitude = Math.Cos(angle);
                    var dCombinedX_dAngle = -magnitude * Math.Sin(angle);
                    var dCombinedX_dWMagnitude = Math.Cos(wAngle);
                    var dCombinedX_dWAngle = -wMagnitude * Math.Sin(wAngle);

                    var dCombinedY_dMagnitude = Math.Sin(angle);
                    var dCombinedY_dAngle = magnitude * Math.Cos(angle);
                    var dCombinedY_dWMagnitude = Math.Sin(wAngle);
                    var dCombinedY_dWAngle = wMagnitude * Math.Cos(wAngle);

                    var combinedMagnitude = Math.Sqrt((combinedX * combinedX) + (combinedY * combinedY));
                    double resultMagnitude = combinedMagnitude * this.weights[i, j];
                    double resultAngle = Math.Atan2(combinedY, combinedX);

                    var dResultMagnitude_dCombinedX = combinedX / combinedMagnitude * this.weights[i, j];
                    var dResultMagnitude_dCombinedY = combinedY / combinedMagnitude * this.weights[i, j];

                    var denominator = ((combinedX * combinedX) + (combinedY * combinedY));
                    var dResultAngle_dCombinedX = -combinedY / denominator;
                    var dResultAngle_dCombinedY = combinedX / denominator;

                    // Changes: Handle separate upstream gradients for both sin and cos branches
                    // This ensures four distinct gradients:
                    // Two for angle and two for wAngle (one for magnitude, one for angle)
                    var upstreamGradientMagnitudeX = dOutput[i, j] * dResultMagnitude_dCombinedX;
                    var upstreamGradientMagnitudeY = dOutput[i, j] * dResultMagnitude_dCombinedY;
                    var upstreamGradientAngleX = dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedX;
                    var upstreamGradientAngleY = dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedY;

                    // Compute gradients for input1 (magnitude and angle)
                    // Instead of combining them into two, we handle four upstream gradients separately.
                    dInput1[i, j] = (upstreamGradientMagnitudeX * dCombinedX_dMagnitude) +
                                    (upstreamGradientMagnitudeY * dCombinedY_dMagnitude) +
                                    (upstreamGradientAngleX * dCombinedX_dMagnitude) +
                                    (upstreamGradientAngleY * dCombinedY_dMagnitude);

                    if (i == 0 && j == 0)
                    {
                        var combined = upstreamGradientMagnitudeX + upstreamGradientAngleX; // -2.652
                        var combined2 = upstreamGradientMagnitudeY + upstreamGradientAngleY; // -1.538
                    }

                    dInput1[i, j + (this.input1.Cols / 2)] = (upstreamGradientMagnitudeX * dCombinedX_dAngle) +
                                                            (upstreamGradientMagnitudeY * dCombinedY_dAngle) +
                                                            (upstreamGradientAngleX * dCombinedX_dAngle) +
                                                            (upstreamGradientAngleY * dCombinedY_dAngle);

                    // Compute gradients for input2 (wMagnitude and wAngle)
                    dInput2[i, j] = (upstreamGradientMagnitudeX * dCombinedX_dWMagnitude) +
                                    (upstreamGradientMagnitudeY * dCombinedY_dWMagnitude) +
                                    (upstreamGradientAngleX * dCombinedX_dWMagnitude) +
                                    (upstreamGradientAngleY * dCombinedY_dWMagnitude);

                    dInput2[i, j + (this.input2.Cols / 2)] = (upstreamGradientMagnitudeX * dCombinedX_dWAngle) +
                                                            (upstreamGradientMagnitudeY * dCombinedY_dWAngle) +
                                                            (upstreamGradientAngleX * dCombinedX_dWAngle) +
                                                            (upstreamGradientAngleY * dCombinedY_dWAngle);

                    // Gradient for the weights
                    var dResultMagnitude_dWeights = combinedMagnitude;
                    dWeights[i, j] = dOutput[i, j] * dResultMagnitude_dWeights;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

        public BackwardResult Backward2(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    var magnitude = this.input1[i, j];
                    var angle = this.input1[i, j + (this.input1.Cols / 2)];
                    var wMagnitude = this.input2[i, j];
                    var wAngle = this.input2[i, j + (this.input2.Cols / 2)];

                    var cAngle = Math.Cos(angle);
                    var sAngle = Math.Sin(angle);
                    var cwAngle = Math.Cos(wAngle);
                    var swAngle = Math.Sin(wAngle);

                    var x1 = magnitude * Math.Cos(angle);
                    var y1 = magnitude * Math.Sin(angle);
                    var x2 = wMagnitude * Math.Cos(wAngle);
                    var y2 = wMagnitude * Math.Sin(wAngle);

                    var combinedX = x1 + x2;
                    var combinedY = y1 + y2;

                    var dCombinedX_dMagnitude = Math.Cos(angle);
                    var dCombinedX_dAngle = -magnitude * Math.Sin(angle);
                    var dCombinedX_dWMagnitude = Math.Cos(wAngle);
                    var dCombinedX_dWAngle = -wMagnitude * Math.Sin(wAngle);

                    var dCombinedY_dMagnitude = Math.Sin(angle);
                    var dCombinedY_dAngle = magnitude * Math.Cos(angle);
                    var dCombinedY_dWMagnitude = Math.Sin(wAngle);
                    var dCombinedY_dWAngle = wMagnitude * Math.Cos(wAngle);

                    var combinedMagnitude = Math.Sqrt((combinedX * combinedX) + (combinedY * combinedY));
                    double resultMagnitude = combinedMagnitude * this.weights[i, j];
                    double resultAngle = Math.Atan2(combinedY, combinedX);

                    var dResultMagnitude_dCombinedX = combinedX / combinedMagnitude * this.weights[i, j];
                    var dResultMagnitude_dCombinedY = combinedY / combinedMagnitude * this.weights[i, j];

                    var denominator = ((combinedX * combinedX) + (combinedY * combinedY));
                    var dResultAngle_dCombinedX = -combinedY / denominator;
                    var dResultAngle_dCombinedY = combinedX / denominator;

                    dInput1[i, j] = (dOutput[i, j] * dResultMagnitude_dCombinedX * dCombinedX_dMagnitude) +
                                    (dOutput[i, j] * dResultMagnitude_dCombinedY * dCombinedY_dMagnitude) +
                                    (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedX * dCombinedX_dMagnitude) +
                                    (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedY * dCombinedY_dMagnitude);

                    dInput1[i, j + (this.input1.Cols / 2)] = (dOutput[i, j] * dResultMagnitude_dCombinedX * dCombinedX_dAngle) +
                                                            (dOutput[i, j] * dResultMagnitude_dCombinedY * dCombinedY_dAngle) +
                                                            (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedX * dCombinedX_dAngle) +
                                                            (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedY * dCombinedY_dAngle);

                    dInput2[i, j] = (dOutput[i, j] * dResultMagnitude_dCombinedX * dCombinedX_dWMagnitude) +
                                    (dOutput[i, j] * dResultMagnitude_dCombinedY * dCombinedY_dWMagnitude) +
                                    (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedX * dCombinedX_dWMagnitude) +
                                    (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedY * dCombinedY_dWMagnitude);

                    dInput2[i, j + (this.input2.Cols / 2)] = (dOutput[i, j] * dResultMagnitude_dCombinedX * dCombinedX_dWAngle) +
                                                            (dOutput[i, j] * dResultMagnitude_dCombinedY * dCombinedY_dWAngle) +
                                                            (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedX * dCombinedX_dWAngle) +
                                                            (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dCombinedY * dCombinedY_dWAngle);

                    var dResultMagnitude_dWeights = Math.Sqrt((combinedX * combinedX) + (combinedY * combinedY));

                    dWeights[i, j] = dOutput[i, j] * dResultMagnitude_dWeights;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }
    }
}