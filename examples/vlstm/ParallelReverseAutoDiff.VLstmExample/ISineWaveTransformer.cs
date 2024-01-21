//------------------------------------------------------------------------------
// <copyright file="ISineWaveTransformer.cs" author="ameritusweb" date="1/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VLstmExample
{
    /// <summary>
    /// A sine wave transformer.
    /// </summary>
    public interface ISineWaveTransformer
    {
        /// <summary>
        /// Transforms the input.
        /// </summary>
        /// <param name="x">The time step.</param>
        /// <returns>The transformed data point.</returns>
        double Transform(double x);

        /// <summary>
        /// Updates the rotation angle.
        /// </summary>
        /// <param name="angleRadians">The rotation angle in radians.</param>
        void UpdateRotationAngle(double angleRadians);
    }
}
