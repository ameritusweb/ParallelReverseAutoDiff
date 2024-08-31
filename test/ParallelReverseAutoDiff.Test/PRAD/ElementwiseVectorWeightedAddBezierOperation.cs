using ParallelReverseAutoDiff.RMAD;
using System.Diagnostics;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    /// <summary>
    /// Element-wise vector weighted add Bezier operation.
    /// </summary>
    public class ElementwiseVectorWeightedAddBezierOperation
    {
        /// <summary>
        /// Performs the forward operation for the element-wise vector summation function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector summation operation.</param>
        /// <param name="input2">The second input to the element-wise vector summation operation.</param>
        /// <param name="weights">The weights for the element-wise vector summation operation.</param>
        /// <param name="alpha">The alpha parameter.</param>
        /// <param name="beta">The beta parameter.</param>
        /// <param name="gamma">The gamma parameter.</param>
        /// <param name="lambda">The lambda parameter.</param>
        /// <param name="N_cos">The N_cos parameter.</param>
        /// <param name="N_sin">The N_sin parameter.</param>
        /// <param name="p0">The p0 parameter.</param>
        /// <param name="p1">The p1 parameter.</param>
        /// <param name="p2">The p2 parameter.</param>
        /// <param name="p3">The p3 parameter.</param>
        /// <returns>The output of the element-wise vector summation operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights,
                      Matrix N_sin, Matrix N_cos,
                      Matrix p0, Matrix p1, Matrix p2,
                      Matrix alpha, Matrix beta, Matrix lambda, Matrix gamma)
        {
            var output = new Matrix(input1.Rows, input1.Cols);

            Parallel.For(0, input1.Rows, new ParallelOptions() { MaxDegreeOfParallelism = 1 }, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    double magnitude1 = input1[i, j];
                    double angle1 = input1[i, j + (input1.Cols / 2)];
                    double magnitude2 = input2[i, j];
                    double angle2 = input2[i, j + (input2.Cols / 2)];

                    var wave1 = BezierWaveform(1, angle1, N_cos[i, j], p0[i, j], p1[i, j], p2[i, j]);

                    double x1 = magnitude1 * wave1;
                    double y1 = magnitude1 * BezierWaveform(2, angle1, N_sin[i, j], p0[i, j], p1[i, j], p2[i, j]);
                    double x2 = magnitude2 * BezierWaveform(3, angle2, N_cos[i, j], p0[i, j], p1[i, j], p2[i, j]);
                    double y2 = magnitude2 * BezierWaveform(4, angle2, N_sin[i, j], p0[i, j], p1[i, j], p2[i, j]);

                    Debug.WriteLine($"i: {i}, j: {j}, x1: {x1}, x2: {x2}");

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    Debug.WriteLine($"i: {i}, j: {j}, sumx: {sumx}, sumy: {sumy}");

                    double resultMagnitude = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    double resultAngle = Atan2Approximation(sumy, sumx,
                                                            alpha[i, j], beta[i, j], lambda[i, j], gamma[i, j],
                                                            N_cos[i, j], N_sin[i, j],
                                                            p0[i, j], p1[i, j], p2[i, j]);

                    Debug.WriteLine($"i: {i}, j: {j}, resultMagnitude: {resultMagnitude}, resultAngle: {resultAngle}");

                    output[i, j] = resultMagnitude;
                    output[i, j + (input1.Cols / 2)] = resultAngle;
                }
            });

            return output;
        }

        private double BezierWaveform(int number, double x, double N, double p0, double p1, double p2)
        {
            // N = 6, x = 70
            double nSquared = N * N; // 36
            double halfNSquared = 0.5 * nSquared; // 18
            double xMod = x % nSquared; // == 70 % 36 == 34
            bool segment = xMod < halfNSquared; // 34 < 18 == false
            double t = xMod / halfNSquared; // 34 / 18 == 1.8888888888888888
            double tMod = t % 1.0; // 0.8888888888888888

            double y1 = ConstrainedBezier(number, tMod, p0, p1, p2);
            double y2 = ConstrainedBezier(number, tMod, p0 * -1d, p1 * -1d, p2 * -1d);

            Debug.WriteLine($"number: {number}, xMod: {xMod}, t: {t}, y1: {y1}, y2: {y2}");

            return segment ? y1 : y2;
        }

        private double CubicBezier(int number, double t, double p0, double p1, double p2, double p3)
        {
            double t2 = t * t;
            double t3 = t2 * t;
            double mt = 1.0 - t;
            double mt2 = mt * mt;
            double mt3 = mt2 * mt;

            var r0 = mt3 * p0;
            var r1 = 3 * t * mt2 * p1;
            var r2 = 3 * t2 * mt * p2;
            var r3 = t3 * p3;

            Debug.WriteLine($"number: {number}, t: {t}, r0: {r0}, r1: {r1}, r2: {r2}, r3: {r3}");

            return r0 +
                   r1 +
                   r2 +
                   r3;
        }

        /// <summary>
        /// Evaluates the constrained Bézier curve with three control points.
        /// </summary>
        /// <param name="number">The number of the control point.</param>
        /// <param name="t">The parameter t, where 0 <= t <= 1.</param>
        /// <param name="p0">The first control point.</param>
        /// <param name="p1">The second control point.</param>
        /// <param name="p2">The third control point.</param>
        /// <returns>The evaluated point on the Bézier curve at parameter t.</returns>
        private double ConstrainedBezier(int number, double t, double p0, double p1, double p2)
        {
            double t2 = t * t;
            double t3 = t2 * t;

            double mt = 1.0 - t;
            double mt2 = mt * mt;
            double mt3 = mt2 * mt;

            // Calculate the contribution from each control point
            var r0 = (4 * mt3 * t) * p0;
            var r1 = (6 * mt2 * t2) * p1;
            var r2 = (4 * mt * t3) * p2;

            Debug.WriteLine($"number: {number}, t: {t}, r0: {r0}, r1: {r1}, r2: {r2}");

            return r0 + r1 + r2;
        }

        private double Atan2Approximation(double y, double x,
                                          double alpha, double beta, double lambda, double gamma,
                                          double N_cos, double N_sin,
                                          double p0, double p1, double p2)
        {
            double BWCosX = BezierWaveform(5, x, N_cos, p0, p1, p2);
            double BWCosY = BezierWaveform(6, y, N_cos, p0, p1, p2);
            double BWSinX = BezierWaveform(7, x, N_sin, p0, p1, p2);
            double BWSinY = BezierWaveform(8, y, N_sin, p0, p1, p2);

            var term1 = alpha * BWCosX;
            var term2 = beta * BWSinX;
            var term3 = lambda * BWCosY;
            var term4 = gamma * BWSinY;

            Debug.WriteLine($"term1: {term1}, term2: {term2}, term3: {term3}, term4: {term4}");

            return term1 + term2 + term3 + term4;
        }
    }
}