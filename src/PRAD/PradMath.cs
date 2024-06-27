//------------------------------------------------------------------------------
// <copyright file="PradMath.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Provides common mathematical functions for PradOp operations.
    /// </summary>
    public static class PradMath
    {
        /// <summary>
        /// PI constant.
        /// </summary>
        public static readonly float PI = (float)Math.PI;

        /// <summary>
        /// Computes the cosine of a double-precision floating-point number.
        /// </summary>
        /// <param name="x">The angle, measured in radians.</param>
        /// <returns>The cosine of x.</returns>
        public static double Cos(double x)
        {
            return Math.Cos(x);
        }

        /// <summary>
        /// Computes the cosine of a single-precision floating-point number.
        /// </summary>
        /// <param name="x">The angle, measured in radians.</param>
        /// <returns>The cosine of x.</returns>
        public static float Cos(float x)
        {
            return (float)Math.Cos(x); // Explicit cast to float
        }

        /// <summary>
        /// Computes the sine of a double-precision floating-point number.
        /// </summary>
        /// <param name="x">The angle, measured in radians.</param>
        /// <returns>The sine of x.</returns>
        public static double Sin(double x)
        {
            return Math.Sin(x);
        }

        /// <summary>
        /// Computes the sine of a single-precision floating-point number.
        /// </summary>
        /// <param name="x">The angle, measured in radians.</param>
        /// <returns>The sine of x.</returns>
        public static float Sin(float x)
        {
            return (float)Math.Sin(x); // Explicit cast to float
        }

        /// <summary>
        /// Computes the exponential of a double-precision floating-point number.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <returns>The exponential of x.</returns>
        public static double Exp(double x)
        {
            return Math.Exp(x);
        }

        /// <summary>
        /// Computes the exponential of a single-precision floating-point number.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <returns>The exponential of x.</returns>
        public static float Exp(float x)
        {
            return (float)Math.Exp(x); // Explicit cast to float
        }

        /// <summary>
        /// Computes the square root of a double-precision floating-point number.
        /// </summary>
        /// <param name="x">The number whose square root is to be found.</param>
        /// <returns>The square root of x.</returns>
        public static double Sqrt(double x)
        {
            return Math.Sqrt(x);
        }

        /// <summary>
        /// Computes the square root of a single-precision floating-point number.
        /// </summary>
        /// <param name="x">The number whose square root is to be found.</param>
        /// <returns>The square root of x.</returns>
        public static float Sqrt(float x)
        {
            return (float)Math.Sqrt(x); // Explicit cast to float
        }

        /// <summary>
        /// Computes the arctangent of the quotient of two double-precision floating-point numbers.
        /// </summary>
        /// <param name="y">The y coordinate of the point (x, y).</param>
        /// <param name="x">The x coordinate of the point (x, y).</param>
        /// <returns>The arctangent of y/x, in radians.</returns>
        public static double Atan2(double y, double x)
        {
            return Math.Atan2(y, x);
        }

        /// <summary>
        /// Computes the arctangent of the quotient of two single-precision floating-point numbers.
        /// </summary>
        /// <param name="y">The y coordinate of the point (x, y).</param>
        /// <param name="x">The x coordinate of the point (x, y).</param>
        /// <returns>The arctangent of y/x, in radians.</returns>
        public static float Atan2(float y, float x)
        {
            return (float)Math.Atan2(y, x); // Explicit cast to float
        }

        /// <summary>
        /// Returns a specified number raised to the specified power.
        /// </summary>
        /// <param name="x">The number to be raised to a power.</param>
        /// <param name="y">The power to which x is to be raised.</param>
        /// <returns>The number x raised to the power y.</returns>
        public static double Pow(double x, double y)
        {
            return Math.Pow(x, y);
        }

        /// <summary>
        /// Returns a specified number raised to the specified power.
        /// </summary>
        /// <param name="x">The number to be raised to a power.</param>
        /// <param name="y">The power to which x is to be raised.</param>
        /// <returns>The number x raised to the power y.</returns>
        public static float Pow(float x, float y)
        {
            return (float)Math.Pow(x, y); // Explicit cast to float
        }

        /// <summary>
        /// Returns the hyperbolic tangent of the specified angle.
        /// </summary>
        /// <param name="x">An angle, measured in radians.</param>
        /// <returns>The hyperbolic tangent of x.</returns>
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        /// <summary>
        /// Returns the hyperbolic tangent of the specified angle.
        /// </summary>
        /// <param name="x">An angle, measured in radians.</param>
        /// <returns>The hyperbolic tangent of x.</returns>
        public static float Tanh(float x)
        {
            return (float)Math.Tanh(x); // Explicit cast to float
        }

        /// <summary>
        /// Returns the natural (base *e*) logarithm of a specified number.
        /// </summary>
        /// <param name="x">A number whose logarithm is to be found.</param>
        /// <returns>The natural logarithm of x.</returns>
        public static double Log(double x)
        {
            return Math.Log(x);
        }

        /// <summary>
        /// Returns the natural (base *e*) logarithm of a specified number.
        /// </summary>
        /// <param name="x">A number whose logarithm is to be found.</param>
        /// <returns>The natural logarithm of x.</returns>
        public static float Log(float x)
        {
            return (float)Math.Log(x); // Explicit cast to float
        }
    }
}
