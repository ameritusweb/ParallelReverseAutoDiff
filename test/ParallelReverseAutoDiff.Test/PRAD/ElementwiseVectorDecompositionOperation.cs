using ParallelReverseAutoDiff.RMAD;
using System.Diagnostics;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    /// <summary>
    /// Element-wise vector projection operation.
    /// </summary>
    public class ElementwiseVectorDecompositionOperation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;

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

            var output = new Matrix(this.input1.Rows, this.input1.Cols * 10);

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];

                    double wMagnitudePivot = input2[i, j * 5];
                    double wAnglePivot = input2[i, (j * 5) + (input2.Cols / 2)];

                    double wMagnitude1 = input2[i, (j * 5) + 1];
                    double wAngle1 = input2[i, (j * 5) + 1 + (input2.Cols / 2)];

                    double wMagnitude2 = input2[i, (j * 5) + 2];
                    double wAngle2 = input2[i, (j * 5) + 2 + (input2.Cols / 2)];

                    double wMagnitude3 = input2[i, (j * 5) + 3];
                    double wAngle3 = input2[i, (j * 5) + 3 + (input2.Cols / 2)];

                    double wMagnitude4 = input2[i, (j * 5) + 4];
                    double wAngle4 = input2[i, (j * 5) + 4 + (input2.Cols / 2)];

                    // Compute vector components
                    double x = magnitude * PradMath.Cos(angle);
                    double y = magnitude * PradMath.Sin(angle);
                    double xPivot = wMagnitudePivot * PradMath.Cos(wAnglePivot);
                    double yPivot = wMagnitudePivot * PradMath.Sin(wAnglePivot);

                    double x1 = wMagnitude1 * PradMath.Cos(wAngle1);
                    double y1 = wMagnitude1 * PradMath.Sin(wAngle1);

                    double x2 = wMagnitude2 * PradMath.Cos(wAngle2);
                    double y2 = wMagnitude2 * PradMath.Sin(wAngle2);

                    double x3 = wMagnitude3 * PradMath.Cos(wAngle3);
                    double y3 = wMagnitude3 * PradMath.Sin(wAngle3);

                    double x4 = wMagnitude4 * PradMath.Cos(wAngle4);
                    double y4 = wMagnitude4 * PradMath.Sin(wAngle4);

                    var weight = this.weights[i, j] + 0.01d;

                    double sumx = (x + xPivot) / (weight + 1E-9);
                    double sumy = (y + yPivot) / (weight + 1E-9);

                    double diffx1 = sumx - x1;
                    double diffy1 = sumy - y1;

                    Debug.WriteLine($"C# Result[{i},{j}]: sumx={sumx}, x1={x1}");

                    double diffx2 = -sumx - x2;
                    double diffy2 = -sumy - y2;

                    double diffx3 = sumx - x3;
                    double diffy3 = sumy - y3;

                    Debug.WriteLine($"C# Result[{i},{j}]: sumx={sumx}, x3={x3}");

                    double diffx4 = -sumx - x4;
                    double diffy4 = -sumy - y4;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude1 = PradMath.Sqrt((diffx1 * diffx1) + (diffy1 * diffy1));
                    double resultAngle1 = PradMath.Atan2(diffy1, diffx1);

                    double resultMagnitude2 = PradMath.Sqrt((diffx2 * diffx2) + (diffy2 * diffy2));
                    double resultAngle2 = PradMath.Atan2(diffy2, diffx2);

                    double resultMagnitude3 = PradMath.Sqrt((diffx3 * diffx3) + (diffy3 * diffy3));
                    double resultAngle3 = PradMath.Atan2(diffy3, diffx3);

                    double resultMagnitude4 = PradMath.Sqrt((diffx4 * diffx4) + (diffy4 * diffy4));
                    double resultAngle4 = PradMath.Atan2(diffy4, diffx4);

                    // Intermediate result debugging
                    Debug.WriteLine($"C# Result[{i},{j}]: m1={resultMagnitude1}, a1={resultAngle1}");
                    Debug.WriteLine($"C# Result[{i},{j}]: m2={resultMagnitude2}, a2={resultAngle2}");
                    Debug.WriteLine($"C# Result[{i},{j}]: m3={resultMagnitude3}, a3={resultAngle3}");
                    Debug.WriteLine($"C# Result[{i},{j}]: m4={resultMagnitude4}, a4={resultAngle4}");
                    //Debug.WriteLine($"C# Result[{i},{j}]: DiffX1={diffx1}, DiffY1={diffy1}");
                    //Debug.WriteLine($"C# Result[{i},{j}]: DiffX2={diffx2}, DiffY2={diffy2}");
                    //Debug.WriteLine($"C# Result[{i},{j}]: DiffX3={diffx3}, DiffY3={diffy3}");
                    //Debug.WriteLine($"C# Result[{i},{j}]: DiffX4={diffx4}, DiffY4={diffy4}");

                    output[i, j * 10] = magnitude;
                    output[i, (j * 10) + (this.input1.Cols * 10 / 2)] = angle;

                    output[i, (j * 10) + 1] = wMagnitudePivot;
                    output[i, (j * 10) + 1 + (this.input1.Cols * 10 / 2)] = wAnglePivot;

                    output[i, (j * 10) + 2] = wMagnitude1;
                    output[i, (j * 10) + 2 + (this.input1.Cols * 10 / 2)] = wAngle1;

                    output[i, (j * 10) + 3] = wMagnitude2;
                    output[i, (j * 10) + 3 + (this.input1.Cols * 10 / 2)] = wAngle2;

                    output[i, (j * 10) + 4] = wMagnitude3;
                    output[i, (j * 10) + 4 + (this.input1.Cols * 10 / 2)] = wAngle3;

                    output[i, (j * 10) + 5] = wMagnitude4;
                    output[i, (j * 10) + 5 + (this.input1.Cols * 10 / 2)] = wAngle4;

                    output[i, (j * 10) + 6] = resultMagnitude1;
                    output[i, (j * 10) + 6 + (this.input1.Cols * 10 / 2)] = resultAngle1;

                    output[i, (j * 10) + 7] = resultMagnitude2;
                    output[i, (j * 10) + 7 + (this.input1.Cols * 10 / 2)] = resultAngle2;

                    output[i, (j * 10) + 8] = resultMagnitude3;
                    output[i, (j * 10) + 8 + (this.input1.Cols * 10 / 2)] = resultAngle3;

                    output[i, (j * 10) + 9] = resultMagnitude4;
                    output[i, (j * 10) + 9 + (this.input1.Cols * 10 / 2)] = resultAngle4;
                }
            });

            return output;
        }
    }
}