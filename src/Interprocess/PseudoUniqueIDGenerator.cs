//------------------------------------------------------------------------------
// <copyright file="PseudoUniqueIDGenerator.cs" author="ameritusweb" date="5/9/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Interprocess
{
    using System;
    using System.Threading;

    /// <summary>
    /// Generates a pseudo-unique ID.
    /// </summary>
    public class PseudoUniqueIDGenerator
    {
        private static readonly Lazy<PseudoUniqueIDGenerator> LazyLoadedInstance = new Lazy<PseudoUniqueIDGenerator>(() => new PseudoUniqueIDGenerator(), true);
        private int currentID;

        /// <summary>
        /// Gets an instance of the <see cref="PseudoUniqueIDGenerator"/> class.
        /// </summary>
        public static PseudoUniqueIDGenerator Instance => LazyLoadedInstance.Value;

        /// <summary>
        /// Gets the next ID.
        /// </summary>
        /// <returns>The next ID.</returns>
        public int GetNextID()
        {
            int nextID = Interlocked.Increment(ref this.currentID);

            // If we reached int.MaxValue, reset back to 0
            if (nextID == int.MaxValue)
            {
                Interlocked.Exchange(ref this.currentID, 0);
            }

            return nextID;
        }
    }
}
