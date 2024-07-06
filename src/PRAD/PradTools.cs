//------------------------------------------------------------------------------
// <copyright file="PradTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;

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
        /// Gets Epsilon.
        /// </summary>
        internal static double Epsilon { get; } = 1e-15;

        /// <summary>
        /// Gets Two.
        /// </summary>
        internal static double Two { get; } = 2.0;

        /// <summary>
        /// Gets Half.
        /// </summary>
        internal static double Half { get; } = 0.5;

        /// <summary>
        /// Gets Negative One.
        /// </summary>
        internal static double NegativeOne { get; } = -1.0;

        /// <summary>
        /// Gets One.
        /// </summary>
        internal static double One { get; } = 1.0;

        /// <summary>
        /// Cast a double to a double.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The double.</returns>
        internal static double Cast(double value)
        {
            return value;
        }

        /// <summary>
        /// Gets the element size of the array.
        /// </summary>
        /// <typeparam name="T">The type of the array.</typeparam>
        /// <param name="array">The array.</param>
        /// <returns>The element size.</returns>
        internal static int GetElementSize<T>(T[] array)
            where T : unmanaged
        {
            return Buffer.ByteLength(array) / array.Length;
        }
    }
}
