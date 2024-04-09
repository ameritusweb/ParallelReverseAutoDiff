//------------------------------------------------------------------------------
// <copyright file="SineWaveVectorGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    /// <summary>
    /// A sine wave generator that also generates vectors representing the slope of the wave at each point.
    /// </summary>
    public class SineWaveVectorGenerator
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SineWaveVectorGenerator"/> class.
        /// </summary>
        /// <param name="amplitude">The amplitude of the sine wave.</param>
        /// <param name="frequency">The frequency of the sine wave.</param>
        /// <param name="phase">The phase of the sine wave.</param>
        public SineWaveVectorGenerator(double amplitude, double frequency, double phase)
        {
            this.Amplitude = amplitude;
            this.Frequency = frequency;
            this.Phase = phase;
        }

        /// <summary>
        /// Gets or sets the amplitude of the sine wave.
        /// </summary>
        public double Amplitude { get; set; }

        /// <summary>
        /// Gets or sets the frequency of the sine wave.
        /// </summary>
        public double Frequency { get; set; }

        /// <summary>
        /// Gets or sets the phase of the sine wave.
        /// </summary>
        public double Phase { get; set; }

        /// <summary>
        /// Generates a sine wave with vectors representing the slope of the wave at each point.
        /// </summary>
        /// <param name="samples">The number of samples.</param>
        /// <param name="cycles">The number of cycles.</param>
        /// <returns>The magnitude and direction.</returns>
        public List<(double Y, (double Magnitude, double Direction) Vector)> GenerateWaveWithVectors(int samples, int cycles = 1)
        {
            var results = new List<(double Y, (double Magnitude, double Direction) Vector)>();

            for (int i = 0; i < samples; i++)
            {
                // Adjusting t to cover 'cycles' number of full sine wave cycles
                double t = (double)i / samples * cycles;
                double y = this.Amplitude * Math.Sin((2 * Math.PI * this.Frequency * t) + this.Phase);
                double slope = 2 * Math.PI * this.Frequency * this.Amplitude * Math.Cos((2 * Math.PI * this.Frequency * t) + this.Phase);
                double direction = Math.Atan(slope); // Angle in radians
                double magnitude = Math.Abs(slope); // Using absolute slope value as magnitude for simplicity

                results.Add((y, (magnitude, direction)));
            }

            return results;
        }
    }
}
