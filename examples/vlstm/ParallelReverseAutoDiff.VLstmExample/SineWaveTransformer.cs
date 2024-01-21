//------------------------------------------------------------------------------
// <copyright file="SineWaveTransformer.cs" author="ameritusweb" date="1/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VLstmExample
{
    /// <summary>
    /// A sine wave transformer.
    /// </summary>
    public class SineWaveTransformer : ISineWaveTransformer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SineWaveTransformer"/> class.
        /// </summary>
        /// <param name="amplitude">The amplitude.</param>
        /// <param name="frequency">The frequency.</param>
        public SineWaveTransformer(double amplitude, double frequency)
        {
            this.Amplitude = amplitude;
            this.Frequency = frequency;
        }

        /// <summary>
        /// Gets or sets the amplitude.
        /// </summary>
        public double Amplitude { get; set; }

        /// <summary>
        /// Gets or sets the frequency.
        /// </summary>
        public double Frequency { get; set; }

        /// <summary>
        /// Gets the rotation angle in radians.
        /// </summary>
        public double RotationAngleRadians { get; private set; }

        /// <summary>
        /// Updates the rotation angle.
        /// </summary>
        /// <param name="angleRadians">The angle in radians.</param>
        public void UpdateRotationAngle(double angleRadians) => this.RotationAngleRadians = angleRadians;

        /// <summary>
        /// Transforms the input.
        /// </summary>
        /// <param name="x">The time step.</param>
        /// <returns>The transformed data point.</returns>
        public double Transform(double x) => this.Amplitude * Math.Sin((this.Frequency * x) + this.RotationAngleRadians);
    }
}
