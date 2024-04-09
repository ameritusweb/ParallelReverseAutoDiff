//------------------------------------------------------------------------------
// <copyright file="IEnumerableExtensions.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VLstmExample.Common
{
    /// <summary>
    /// Provides extension methods for <see cref="IEnumerable{T}"/>.
    /// </summary>
    public static class IEnumerableExtensions
    {
        /// <summary>
        /// Returns the item and its index.
        /// </summary>
        /// <typeparam name="T">The type of item.</typeparam>
        /// <param name="source">The source enumerable.</param>
        /// <returns>The target enumerable.</returns>
        public static IEnumerable<(T Item, int Index)> WithIndex<T>(this IEnumerable<T> source)
        {
            int index = 0;
            foreach (var item in source)
            {
                yield return (item, index);
                index++;
            }
        }

        /// <summary>
        /// Returns the item and the next item.
        /// </summary>
        /// <typeparam name="T">The type of the item.</typeparam>
        /// <param name="source">The source enumerable.</param>
        /// <returns>The target enumerable.</returns>
        public static IEnumerable<(T Current, T? Next)> WithNext<T>(this IEnumerable<T> source)
        {
            using (var enumerator = source.GetEnumerator())
            {
                if (!enumerator.MoveNext())
                {
                    yield break;
                }

                var current = enumerator.Current;

                while (enumerator.MoveNext())
                {
                    yield return (current, enumerator.Current);
                    current = enumerator.Current;
                }

                yield return (current, default);
            }
        }
    }
}
