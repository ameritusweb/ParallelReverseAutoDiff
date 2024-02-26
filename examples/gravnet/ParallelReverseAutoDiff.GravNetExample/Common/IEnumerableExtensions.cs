﻿//------------------------------------------------------------------------------
// <copyright file="IEnumerableExtensions.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GravNetExample.Common
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

        public static void WithRepeat<T>(this IEnumerable<T> source, Action<T, RepeatToken> action)
        {
            var repeatToken = new RepeatToken();

            foreach (var item in source)
            {
                do
                {
                    repeatToken.Decrement();
                    action(item, repeatToken);
                }
                while (repeatToken.ShouldRepeat);
            }
        }

        public static async Task WithRepeatAsync<T>(this IEnumerable<T> source, Func<T, RepeatToken, Task> action)
        {
            foreach (var item in source)
            {
                var repeatToken = new RepeatToken();
                do
                {
                    repeatToken.Decrement();
                    await action(item, repeatToken);

                    if (repeatToken.ShouldContinue)
                    {
                        break; // Exit the current iteration immediately
                    }
                }
                while (repeatToken.ShouldRepeat);
            }
        }

        public static double[,] To2D(this IEnumerable<double[]> enumerable)
        {
            var list = enumerable.ToList();

            if (!list.Any())
            {
                return new double[0, 0]; // Return an empty 2D array if the input is empty
            }

            int rows = list.Count;
            int cols = list[0].Length;

            double[,] twoDArray = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    twoDArray[i, j] = list[i][j];
                }
            }

            return twoDArray;
        }
    }
}
