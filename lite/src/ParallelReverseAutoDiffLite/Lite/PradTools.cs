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
        internal static float Zero { get; } = 0.0f;

        /// <summary>
        /// Gets Epsilon.
        /// </summary>
        internal static float Epsilon { get; } = 1e-15f;

        /// <summary>
        /// Gets Two.
        /// </summary>
        internal static float Two { get; } = 2.0f;

        /// <summary>
        /// Gets Half.
        /// </summary>
        internal static float Half { get; } = 0.5f;

        /// <summary>
        /// Gets Negative One.
        /// </summary>
        internal static float NegativeOne { get; } = -1.0f;

        /// <summary>
        /// Gets One.
        /// </summary>
        internal static float One { get; } = 1.0f;

        /// <summary>
        /// Gets the SizeOf.
        /// </summary>
        internal static int SizeOf { get; } = sizeof(float);

        /// <summary>
        /// Cast a double to a float.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The float.</returns>
        internal static float Cast(double value)
        {
            return (float)value;
        }

        /// <summary>
        /// Gets an array of floats.
        /// </summary>
        /// <param name="length">The length of the array.</param>
        /// <returns>The allocated array.</returns>
        internal static float[] AllocateArray(int length)
        {
            return new float[length];
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
