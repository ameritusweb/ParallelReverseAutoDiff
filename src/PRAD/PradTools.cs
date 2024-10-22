//------------------------------------------------------------------------------
// <copyright file="PradTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Numerics;

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
        /// Gets One Hundredth.
        /// </summary>
        internal static double OneHundredth { get; } = 0.01;

        /// <summary>
        /// Gets Epsilon.
        /// </summary>
        internal static double Epsilon { get; } = 1e-15;

        /// <summary>
        /// Gets Epsilon9.
        /// </summary>
        internal static double Epsilon9 { get; } = 1e-9;

        /// <summary>
        /// Gets Epsilon10.
        /// </summary>
        internal static double Epsilon10 { get; } = 1e-10;

        /// <summary>
        /// Gets Two.
        /// </summary>
        internal static double Two { get; } = 2.0;

        /// <summary>
        /// Gets Six.
        /// </summary>
        internal static double Six { get; } = 6.0;

        /// <summary>
        /// Gets Negative Two.
        /// </summary>
        internal static double NegativeTwo { get; } = -2.0;

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
        /// Gets the SizeOf.
        /// </summary>
        internal static int SizeOf { get; } = sizeof(double);

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
        /// Gets an array of doubles.
        /// </summary>
        /// <param name="length">The length of the array.</param>
        /// <returns>The allocated array.</returns>
        internal static double[] AllocateArray(int length)
        {
            return new double[length];
        }

        /// <summary>
        /// Gets an array of doubles.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <returns>The array.</returns>
        internal static double[] OneArray(int length)
        {
            var array = new double[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = 1.0;
            }

            return array;
        }

        /// <summary>
        /// Fills an array.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <param name="value">The value.</param>
        /// <returns>The filled array.</returns>
        internal static double[] FillArray(int length, double value)
        {
            var array = new double[length];
            Array.Fill(array, value);
            return array;
        }

        /// <summary>
        /// Gets the vector count.
        /// </summary>
        /// <returns>The vector count.</returns>
        internal static int VectorCount()
        {
            return Vector<double>.Count;
        }

        /// <summary>
        /// Gets the vector zero.
        /// </summary>
        /// <returns>The vector zero.</returns>
        internal static Vector<double> VectorZero()
        {
            return Vector<double>.Zero;
        }

        /// <summary>
        /// Gets the vector one.
        /// </summary>
        /// <returns>The vector one.</returns>
        internal static Vector<double> VectorOne()
        {
            return Vector<double>.One;
        }

        /// <summary>
        /// Gets a vector of doubles.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="index">The index.</param>
        /// <returns>The vector of doubles.</returns>
        internal static Vector<double> AllocateVector(double[] data, int index)
        {
            return new Vector<double>(data, index);
        }

        /// <summary>
        /// Gets a vector of doubles.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The vector of doubles.</returns>
        internal static Vector<double> AllocateVector(double[] data)
        {
            return new Vector<double>(data);
        }

        /// <summary>
        /// Gets a span of doubles.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The span of doubles.</returns>
        internal static Span<double> AllocateSpan(double[] data)
        {
            return new Span<double>(data);
        }

        /// <summary>
        /// Gets a span of doubles.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="start">The start.</param>
        /// <param name="length">The length.</param>
        /// <returns>The span of doubles.</returns>
        internal static Span<double> AllocateSpan(double[] data, int start, int length)
        {
            return new Span<double>(data, start, length);
        }

        /// <summary>
        /// Allocates a vector.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The vector.</returns>
        internal static Vector<double> AllocateVector(double value)
        {
            return new Vector<double>(value);
        }

        /// <summary>
        /// Converts a vector of longs to a vector of doubles.
        /// </summary>
        /// <param name="vector">The vector.</param>
        /// <returns>The converted vector.</returns>
        internal static Vector<double> Convert(Vector<long> vector)
        {
            return Vector.ConvertToDouble(vector);
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
