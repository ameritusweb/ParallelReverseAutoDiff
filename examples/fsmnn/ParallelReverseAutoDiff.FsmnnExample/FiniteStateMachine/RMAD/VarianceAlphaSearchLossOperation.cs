//------------------------------------------------------------------------------
// <copyright file="VarianceAlphaSearchLossOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Variance alpha search loss operation.
    /// </summary>
    public class VarianceAlphaSearchLossOperation
    {
        private const double Epsilon = 1E-9d;
        private const double RefinementRatio = 0.1;
        private const int MaxRecursionDepth = 0;
        private Matrix predicted;
        private double Alpha = 1d;
        private int targetIndex;
        private double trueVariance;
        private double mean;
        private double variance;
        private double maxMin;
        private double trueMaxMin;

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="predicted">1xN matrix of predicted probabilities.</param>
        /// <param name="trueVariance">A scalar value for the true variance.</param>
        /// <param name="trueMaxMin">A scalar value for the true max-min.</param>
        /// <returns>The computed loss.</returns>
        public double Forward(Matrix predicted, double trueVariance, double trueMaxMin)
        {
            // Initialize member variables
            this.predicted = predicted;
            this.trueVariance = trueVariance;
            this.trueMaxMin = trueMaxMin;
            this.targetIndex = -1;

            // Compute the mean and variance of the predicted probabilities
            double mean = predicted[0].Average();
            double max = predicted[0].Max();
            double min = predicted[0].Min();
            double dist = max - min;
            this.maxMin = dist;
            this.mean = mean;
            double variance = predicted[0].Select(x => (x - mean) * (x - mean)).Average();
            this.variance = variance;
            Console.WriteLine("Variance: " + variance + " " + dist);

            double maxDist = double.MinValue;
            int target = -1;
            if (dist < trueMaxMin)
            {
                for (int i = 0; i < predicted.Cols; ++i)
                {
                    if ((predicted[0][i] - mean) > maxDist)
                    {
                        maxDist = predicted[0][i] - mean;
                        target = i;
                    }
                }
            }
            else if (dist > trueMaxMin)
            {
                for (int i = 0; i < predicted.Cols; ++i)
                {
                    if ((mean - predicted[0][i]) > maxDist)
                    {
                        maxDist = mean - predicted[0][i];
                        target = i;
                    }
                }
            }

            this.targetIndex = target;

            // Compute the final loss based on variance
            double loss = Math.Pow(variance - trueVariance, 2);

            // return loss;
            return dist;
        }

        /// <summary>
        /// Computes the gradient with respect to the predicted probabilities.
        /// </summary>
        /// <returns>1xN matrix of gradients.</returns>
        public Matrix Backward()
        {
            double bestAngle = double.MaxValue;
            double bestAlpha = this.Alpha;  // Store the initial value
            Matrix bestGradient = new Matrix(1, this.predicted.Cols);

            this.RecursiveAlphaSearch(new double[] { 50d }, ref bestAlpha, ref bestAngle, ref bestGradient, MaxRecursionDepth);

            this.Alpha = bestAlpha;
            return bestGradient;
        }

        private void RecursiveAlphaSearch(double[] alphaCandidates, ref double bestAlpha, ref double bestAngle, ref Matrix bestGradient, int recursionDepth)
        {
            foreach (double alphaCandidate in alphaCandidates)
            {
                (Matrix TempGradient, List<double> HessianDiag) tempGradientAndHessianDiag = this.ComputeGradient(alphaCandidate);
                double angle = this.ComputeAngle(tempGradientAndHessianDiag.TempGradient, tempGradientAndHessianDiag.HessianDiag);

                if (Math.Abs(90 - angle) < Math.Abs(90 - bestAngle))
                {
                    bestAlpha = alphaCandidate;
                    bestAngle = angle;
                    bestGradient = tempGradientAndHessianDiag.TempGradient;
                }
            }

            int bestCandidateIndex = Array.IndexOf(alphaCandidates, bestAlpha);

            if (bestCandidateIndex == -1)
            {
                return;
            }

            double stepSize;
            if (bestCandidateIndex == 0)
            {
                if (bestAngle > 90)
                {
                    stepSize = alphaCandidates[0];
                }
                else
                {
                    stepSize = alphaCandidates[1] - alphaCandidates[0];
                }
            }
            else if (bestCandidateIndex == alphaCandidates.Length - 1)
            {
                stepSize = alphaCandidates[bestCandidateIndex] - alphaCandidates[bestCandidateIndex - 1];
            }
            else
            {
                stepSize = alphaCandidates[bestCandidateIndex + 1] - alphaCandidates[bestCandidateIndex];
            }

            if (recursionDepth > 0)
            {
                double newStepSize = stepSize * RefinementRatio;

                // Generate new candidates centered around bestAlpha with the new step size
                double[] newAlphaCandidates = this.GenerateAlphaCandidates(bestAlpha, newStepSize, stepSize);

                this.RecursiveAlphaSearch(newAlphaCandidates, ref bestAlpha, ref bestAngle, ref bestGradient, recursionDepth - 1);
            }
        }

        private double[] GenerateAlphaCandidates(double center, double stepSize, double range)
        {
            List<double> candidates = new List<double>();

            double start = center - (range / 2);
            double end = center + (range / 2);

            for (double alpha = start; alpha <= end; alpha += stepSize)
            {
                candidates.Add(alpha);
            }

            return candidates.ToArray();
        }

        private (Matrix Gradient, List<double> HessianDiag) ComputeGradient(double alphaCandidate)
        {
            Matrix tempGradient = new Matrix(1, this.predicted.Cols);
            List<double> hessianDiag = new List<double>();

            for (int i = 0; i < this.predicted.Cols; i++)
            {
                // Calculate Hessian diagonal term for current prediction
                double hessianTerm = this.targetIndex == i ? alphaCandidate * (-2 / Math.Pow(this.predicted[0][i] + Epsilon, 3)) : 0.0;
                hessianDiag.Add(hessianTerm);

                // Compute gradient term with current alphaCandidate
                tempGradient[0][i] = ((4.0 / this.predicted.Cols) *
                    (this.maxMin - this.trueMaxMin) *
                    (this.predicted[0][i] - this.mean)) + (alphaCandidate * (this.targetIndex == i ? -1 / (this.predicted[0][i] + Epsilon) : 0d));
            }

            return (tempGradient, hessianDiag);
        }

        private double ComputeAngle(Matrix gradient, List<double> hessianDiag)
        {
            // If there's no targetIndex, return a default value
            if (this.targetIndex == -1)
            {
                return double.MaxValue;
            }

            // Identify the non-zero eigenvalue from the Hessian's diagonal
            double eigenvalue = hessianDiag[this.targetIndex];

            // Calculate the shallowest descent direction based on the eigenvalue
            int shallowestDescentIdx = eigenvalue < 0 ? this.targetIndex : -1;

            // Compute the angle between steepest and shallowest descent
            double angle = 0.0;
            if (shallowestDescentIdx != -1)
            {
                double dotProduct = gradient[0][shallowestDescentIdx];
                double normGradient = Math.Sqrt(gradient[0].Sum(val => val * val));
                double cosTheta = dotProduct / normGradient;
                angle = Math.Acos(cosTheta) * (180.0 / Math.PI);
            }

            return angle;
        }
    }
}
