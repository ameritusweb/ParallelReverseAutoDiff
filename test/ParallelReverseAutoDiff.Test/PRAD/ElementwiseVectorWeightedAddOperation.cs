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
    }
}