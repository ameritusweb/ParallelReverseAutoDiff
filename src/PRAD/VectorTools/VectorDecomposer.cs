//------------------------------------------------------------------------------
// <copyright file="VectorDecomposer.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Collections.Generic;
    using System.Numerics;

    /// <summary>
    /// A vector decomposer.
    /// </summary>
    public class VectorDecomposer
    {
        private Random random;

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorDecomposer"/> class.
        /// </summary>
        /// <param name="seed">The seed.</param>
        public VectorDecomposer(int seed = 42)
        {
            this.random = new Random(seed);
        }

        /// <summary>
        /// Decomposes a vector into parts.
        /// </summary>
        /// <param name="targetX">The target X in Cartesian.</param>
        /// <param name="targetY">The target Y in Cartesian.</param>
        /// <param name="n">The number of vectors.</param>
        /// <returns>The list of decomposed vectors.</returns>
        public List<Vector2> DecomposeVector(double targetX, double targetY, int n)
        {
            if (n <= 1)
            {
                // Convert target point to polar coordinates
                double r = Math.Sqrt((targetX * targetX) + (targetY * targetY));
                double theta = Math.Atan2(targetY, targetX);
                return new List<Vector2> { new Vector2((float)r, (float)theta) };
            }

            var result = new List<Vector2>();

            // First, split into two parts with random ratio
            int firstPart = this.random.Next(1, n);
            int secondPart = n - firstPart;

            // Add some randomness to the split point
            double splitRatio = 0.3 + (this.random.NextDouble() * 0.4); // Random value between 0.3 and 0.7

            // Calculate intermediate point
            double midX = targetX * splitRatio;
            double midY = targetY * splitRatio;

            // Add some noise to the intermediate point
            double noiseRadius = Math.Sqrt((targetX * targetX) + (targetY * targetY)) * 0.2; // 20% of total magnitude
            double noiseAngle = this.random.NextDouble() * 2 * Math.PI;
            midX += noiseRadius * Math.Cos(noiseAngle);
            midY += noiseRadius * Math.Sin(noiseAngle);

            // Recursively decompose both parts
            result.AddRange(this.DecomposeVector(midX, midY, firstPart));
            result.AddRange(this.DecomposeVector(targetX - midX, targetY - midY, secondPart));

            return result;
        }

        /// <summary>
        /// Sum vectors together.
        /// </summary>
        /// <param name="vectors">The vectors.</param>
        /// <returns>The sum.</returns>
        public (double X, double Y) SumVectors(List<Vector2> vectors)
        {
            double totalX = 0;
            double totalY = 0;

            foreach (var vector in vectors)
            {
                var res = vector.ToCartesian();
                totalX += res.X;
                totalY += res.Y;
            }

            return (totalX, totalY);
        }

        /// <summary>
        /// Randomize the order of vectors and convert to interleaved array of doubles.
        /// </summary>
        /// <param name="vectors">The list of vectors.</param>
        /// <returns>The array.</returns>
        public double[] RandomizeAndFlatten(List<Vector2> vectors)
        {
            // First randomize the order
            int n = vectors.Count;
            List<Vector2> randomized = new List<Vector2>(vectors);
            for (int i = n - 1; i > 0; i--)
            {
                int j = this.random.Next(i + 1);
                (randomized[i], randomized[j]) = (randomized[j], randomized[i]);
            }

            // Convert to interleaved array: [r1, r2, ..., rn, θ1, θ2, ..., θn]
            double[] result = new double[n * 2];
            for (int i = 0; i < n; i++)
            {
                result[i] = randomized[i].X;               // First half: magnitudes
                result[i + n] = randomized[i].Y;       // Second half: angles
            }

            return result;
        }
    }
}
