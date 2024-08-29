using ParallelReverseAutoDiff.RMAD;
using System.Diagnostics;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    /// <summary>
    /// Element-wise vector projection operation.
    /// </summary>
    public class ElementwiseVectorCartesianSummationOperation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;

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

            var output = new Matrix(1, 2);

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

                    Debug.WriteLine($"Magnitude: {magnitude}, Angle: {angle}, WMagnitude: {wMagnitude}, WAngle: {wAngle}");

                    // Compute vector components
                    double x1 = magnitude * PradMath.Cos(angle);
                    double y1 = magnitude * PradMath.Sin(angle);
                    double x2 = wMagnitude * PradMath.Cos(wAngle);
                    double y2 = wMagnitude * PradMath.Sin(wAngle);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = PradMath.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    double resultAngle = PradMath.Atan2(sumy, sumx);

                    resultVectors[(i * (input1.Cols / 2)) + j, 0] = resultMagnitude;
                    resultVectors[(i * (input1.Cols / 2)) + j, 1] = resultAngle;

                    double localSumX = resultMagnitude * PradMath.Cos(resultAngle);
                    double localSumY = resultMagnitude * PradMath.Sin(resultAngle);

                    Debug.WriteLine($"Resultant Magnitude: {resultMagnitude}, Resultant Angle: {resultAngle}, LocalSumX: {localSumX}, LocalSumY: {localSumY}");

                    sumX += localSumX;
                    sumY += localSumY;
                }

                summationX[i] = sumX;
                summationY[i] = sumY;
            });

            output[0, 0] = summationX.Sum();
            output[0, 1] = summationY.Sum();

            return output;
        }
    }
}