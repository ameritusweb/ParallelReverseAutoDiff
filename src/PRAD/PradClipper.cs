//------------------------------------------------------------------------------
// <copyright file="PradClipper.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// Clips the gradients.
    /// </summary>
    public static class PradClipper
    {
        /// <summary>
        /// Creates a gradient clipper.
        /// </summary>
        /// <param name="clipValue">The clip value.</param>
        /// <returns>The clipper.</returns>
        public static IClipper CreateGradientClipper(double clipValue = 4)
        {
            return new GradientClipper(clipValue);
        }
    }
}
