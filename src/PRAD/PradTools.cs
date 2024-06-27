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
        internal static double Zero { get; } = 0.0;

        /// <summary>
        /// Cast a double to a double.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The double.</returns>
        internal static double Cast(double value)
        {
            return value;
        }
    }
}
