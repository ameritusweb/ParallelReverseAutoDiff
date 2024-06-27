//------------------------------------------------------------------------------
// <copyright file="PradConfig.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// Defines the floating point mode.
    /// </summary>
    internal enum FloatingPointMode
    {
        /// <summary>
        /// Single precision.
        /// </summary>
        Single,

        /// <summary>
        /// Double precision.
        /// </summary>
        Double,
    }

    /// <summary>
    /// PRAD Config.
    /// </summary>
    public static class PradConfig
    {
        /// <summary>
        /// The floating point mode.
        /// </summary>
        internal static readonly FloatingPointMode Mode = FloatingPointMode.Single;
    }
}