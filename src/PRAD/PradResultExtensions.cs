//------------------------------------------------------------------------------
// <copyright file="PradResultExtensions.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;

    /// <summary>
    /// PRAD result extensions.
    /// </summary>
    public static class PradResultExtensions
    {
        /// <summary>
        /// Applies a function to an array of PradResults.
        /// </summary>
        /// <param name="results">An array of PradResults to operate on.</param>
        /// <param name="operation">A function that takes an array of PradResults and returns a single PradResult.</param>
        /// <returns>The result of applying the operation to the array of PradResults.</returns>
        public static PradResult Then(this PradResult[] results, Func<PradResult[], PradResult> operation)
        {
            return operation(results);
        }

        /// <summary>
        /// Applies a function to an array of PradResults that returns multiple PradResults.
        /// </summary>
        /// <param name="results">An array of PradResults to operate on.</param>
        /// <param name="operation">A function that takes an array of PradResults and returns an array of PradResults.</param>
        /// <returns>An array of PradResults from applying the operation to the input array.</returns>
        public static PradResult[] Then(this PradResult[] results, Func<PradResult[], PradResult[]> operation)
        {
            return operation(results);
        }
    }
}
