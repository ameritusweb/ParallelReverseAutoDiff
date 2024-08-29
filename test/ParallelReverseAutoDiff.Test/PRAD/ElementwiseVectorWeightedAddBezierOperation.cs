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
                      Matrix p0, Matrix p1, Matrix p2, Matrix p3,
                      Matrix alpha, Matrix beta, Matrix lambda, Matrix gamma)
        {
            var output = new Matrix(input1.Rows, input1.Cols);

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    double magnitude1 = input1[i, j];
                    double angle1 = input1[i, j + (input1.Cols / 2)];
                    double magnitude2 = input2[i, j];
                    double angle2 = input2[i, j + (input2.Cols / 2)];

                    double x1 = magnitude1 * BezierWaveform(angle1, N_cos[i, j], p0[i, j], p1[i, j], p2[i, j], p3[i, j]);
                    double y1 = magnitude1 * BezierWaveform(angle1, N_sin[i, j], p0[i, j], p1[i, j], p2[i, j], p3[i, j]);
                    double x2 = magnitude2 * BezierWaveform(angle2, N_cos[i, j], p0[i, j], p1[i, j], p2[i, j], p3[i, j]);
                    double y2 = magnitude2 * BezierWaveform(angle2, N_sin[i, j], p0[i, j], p1[i, j], p2[i, j], p3[i, j]);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    double resultMagnitude = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    double resultAngle = Atan2Approximation(sumy, sumx,
                                                            alpha[i, j], beta[i, j], lambda[i, j], gamma[i, j],
                                                            N_cos[i, j], N_sin[i, j],
                                                            p0[i, j], p1[i, j], p2[i, j], p3[i, j]);

                    output[i, j] = resultMagnitude;
                    output[i, j + (input1.Cols / 2)] = resultAngle;
                }
            });

            return output;
        }

        private double BezierWaveform(double x, double N, double p0, double p1, double p2, double p3)
        {
            double nSquared = N * N;
            double halfNSquared = 0.5 * nSquared;
            double xMod = x % nSquared;
            bool segment = xMod < halfNSquared;
            double t = xMod / halfNSquared;

            double y1 = CubicBezier(t, p0, p1, p2, p3);
            double tReflected = -(t - 1.0);
            double y2 = CubicBezier(tReflected, p3, p2, p1, p0);

            return segment ? y1 : y2;
        }

        private double CubicBezier(double t, double p0, double p1, double p2, double p3)
        {
            double t2 = t * t;
            double t3 = t2 * t;
            double mt = 1.0 - t;
            double mt2 = mt * mt;
            double mt3 = mt2 * mt;

            return mt3 * p0 +
                   3 * t * mt2 * p1 +
                   3 * t2 * mt * p2 +
                   t3 * p3;
        }

        private double Atan2Approximation(double y, double x,
                                          double alpha, double beta, double lambda, double gamma,
                                          double N_cos, double N_sin,
                                          double p0, double p1, double p2, double p3)
        {
            double BWCosX = BezierWaveform(x, N_cos, p0, p1, p2, p3);
            double BWCosY = BezierWaveform(y, N_cos, p0, p1, p2, p3);
            double BWSinX = BezierWaveform(x, N_sin, p0, p1, p2, p3);
            double BWSinY = BezierWaveform(y, N_sin, p0, p1, p2, p3);

            return alpha * BWCosX + beta * BWSinX + lambda * BWCosY + gamma * BWSinY;
        }
    }
}