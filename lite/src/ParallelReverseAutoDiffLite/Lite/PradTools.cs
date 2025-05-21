//------------------------------------------------------------------------------
// <copyright file="PradTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Numerics;

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
        /// Gets Epsilon9.
        /// </summary>
        internal static float Epsilon9 { get; } = 1e-9f;

        /// <summary>
        /// Gets Epsilon10.
        /// </summary>
        internal static float Epsilon10 { get; } = 1e-10f;

        /// <summary>
        /// Gets Two.
        /// </summary>
        internal static float Two { get; } = 2.0f;

        /// <summary>
        /// Gets Six.
        /// </summary>
        internal static float Six { get; } = 6.0f;

        /// <summary>
        /// Gets Negative Two.
        /// </summary>
        internal static float NegativeTwo { get; } = -2.0f;

        /// <summary>
        /// Gets Half.
        /// </summary>
        internal static float Half { get; } = 0.5f;

        /// <summary>
        /// Gets One Hundredth.
        /// </summary>
        internal static float OneHundredth { get; } = 0.01f;

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
        /// Cast a double array to a float array.
        /// </summary>
        /// <param name="dArray">The value.</param>
        /// <returns>The float array.</returns>
        internal static float[] Cast(double[] dArray)
        {
            return dArray.Select(x => (float)x).ToArray();
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
        /// Gets the vector count.
        /// </summary>
        /// <returns>The vector count.</returns>
        internal static int VectorCount()
        {
            return Vector<float>.Count;
        }

        /// <summary>
        /// Gets vector zero.
        /// </summary>
        /// <returns>Vector zero.</returns>
        internal static Vector<float> VectorZero()
        {
            return Vector<float>.Zero;
        }

        /// <summary>
        /// Gets vector one.
        /// </summary>
        /// <returns>Vector one.</returns>
        internal static Vector<float> VectorOne()
        {
            return Vector<float>.One;
        }

        /// <summary>
        /// Gets a vector of floats.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="index">The index.</param>
        /// <returns>The vector of floats.</returns>
        internal static Vector<float> AllocateVector(float[] data, int index)
        {
            return new Vector<float>(data, index);
        }

        /// <summary>
        /// Gets a vector of floats.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The vector of floats.</returns>
        internal static Vector<float> AllocateVector(float[] data)
        {
            return new Vector<float>(data);
        }

        /// <summary>
        /// Gets a span of floats.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The span of floats.</returns>
        internal static Span<float> AllocateSpan(float[] data)
        {
            return new Span<float>(data);
        }

        /// <summary>
        /// Gets a span of floats.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="start">The start.</param>
        /// <param name="length">The length.</param>
        /// <returns>The span of floats.</returns>
        internal static Span<float> AllocateSpan(float[] data, int start, int length)
        {
            return new Span<float>(data, start, length);
        }

        /// <summary>
        /// Allocates a vector.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The vector.</returns>
        internal static Vector<float> AllocateVector(float value)
        {
            return new Vector<float>(value);
        }

        /// <summary>
        /// Converts a vector of longs to a vector of floats.
        /// </summary>
        /// <param name="vector">The vector.</param>
        /// <returns>The converted vector.</returns>
        internal static Vector<float> Convert(Vector<int> vector)
        {
            return Vector.ConvertToSingle(vector);
        }

        /// <summary>
        /// Gets an array of floats.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <returns>The array.</returns>
        internal static float[] OneArray(int length)
        {
            var array = new float[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = 1.0f;
            }

            return array;
        }

        /// <summary>
        /// Fills an array.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <param name="value">The value.</param>
        /// <returns>The filled array.</returns>
        internal static float[] FillArray(int length, double value)
        {
            var array = new float[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = (float)value;
            }

            return array;
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
