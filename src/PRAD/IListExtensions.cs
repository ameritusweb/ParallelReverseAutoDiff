//------------------------------------------------------------------------------
// <copyright file="IListExtensions.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// IList Extensions.
    /// </summary>
    public static class IListExtensions
    {
        /// <summary>
        /// Deconstructs the list into first, second and rest.
        /// </summary>
        /// <typeparam name="T">Deconstruct.</typeparam>
        /// <param name="list">List.</param>
        /// <param name="first">First.</param>
        /// <param name="second">Second.</param>
        public static void Deconstruct<T>(this IList<T> list, out T first, out T second)
        {
#pragma warning disable CS8601 // Possible null reference assignment.
            first = list?.Count > 0 ? list![0] : default(T); // or throw
            second = list?.Count > 1 ? list![1] : default(T); // or throw
#pragma warning restore CS8601 // Possible null reference assignment.
        }
    }
}
