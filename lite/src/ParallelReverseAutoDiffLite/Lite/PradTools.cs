//------------------------------------------------------------------------------
// <copyright file="PradTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// PRAD tools.
    /// </summary>
    public static class PradTools
    {
        /// <summary>
        /// Gets Zero.
        /// </summary>
        internal static float Zero { get; } = 0.0f;

        /// <summary>
        /// Cast a double to a float.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The float.</returns>
        internal static float Cast(double value)
        {
            return (float)value;
        }
    }
}
