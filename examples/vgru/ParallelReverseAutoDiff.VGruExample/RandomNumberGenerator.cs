//------------------------------------------------------------------------------
// <copyright file="RandomNumberGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    /// <summary>
    /// A random number generator.
    /// </summary>
    public class RandomNumberGenerator
    {
        private readonly Random random;

        /// <summary>
        /// Initializes a new instance of the <see cref="RandomNumberGenerator"/> class.
        /// </summary>
        public RandomNumberGenerator()
        {
            this.random = new Random(Guid.NewGuid().GetHashCode());
        }

        /// <summary>
        /// Gets a random number between the specified minimum and maximum values.
        /// </summary>
        /// <param name="minValue">The min value.</param>
        /// <param name="maxValue">The max value.</param>
        /// <returns>The random number.</returns>
        public double GetRandomNumber(double minValue, double maxValue)
        {
            return (this.random.NextDouble() * (maxValue - minValue)) + minValue;
        }
    }
}
