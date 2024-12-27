//------------------------------------------------------------------------------
// <copyright file="JsSet.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.SpatialProbabilityNetwork
{
    using System.Collections.Generic;
    using System.Threading;

    /// <summary>
    /// Thread-safe JavaScript-like Set implementation using a HashSet.
    /// </summary>
    public class JsSet : IEnumerable<object>
    {
        private readonly HashSet<object> set = new HashSet<object>();
        private readonly ReaderWriterLockSlim slimLock = new ReaderWriterLockSlim();

        /// <summary>
        /// Adds an item to the set. If the item already exists, it will not be added again.
        /// </summary>
        /// <param name="item">The item to add to the set.</param>
        public void Add(object item)
        {
            this.slimLock.EnterWriteLock();
            try
            {
                this.set.Add(item);
            }
            finally
            {
                this.slimLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Checks if the set contains the specified item.
        /// </summary>
        /// <param name="item">The item to check for existence in the set.</param>
        /// <returns><c>true</c> if the item exists in the set; otherwise, <c>false</c>.</returns>
        public bool Has(object item)
        {
            this.slimLock.EnterReadLock();
            try
            {
                return this.set.Contains(item);
            }
            finally
            {
                this.slimLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Removes the specified item from the set.
        /// </summary>
        /// <param name="item">The item to remove.</param>
        /// <returns><c>true</c> if the item was successfully removed; otherwise, <c>false</c>.</returns>
        public bool Delete(object item)
        {
            this.slimLock.EnterWriteLock();
            try
            {
                return this.set.Remove(item);
            }
            finally
            {
                this.slimLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Clears all items from the set.
        /// </summary>
        public void Clear()
        {
            this.slimLock.EnterWriteLock();
            try
            {
                this.set.Clear();
            }
            finally
            {
                this.slimLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Returns all items in the set as an enumerable collection.
        /// </summary>
        /// <returns>An enumerable list containing all items in the set.</returns>
        public IEnumerable<object> Entries()
        {
            this.slimLock.EnterReadLock();
            try
            {
                return new List<object>(this.set);
            }
            finally
            {
                this.slimLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Returns an enumerator that iterates through the set.
        /// </summary>
        /// <returns>An enumerator for the set.</returns>
        public IEnumerator<object> GetEnumerator()
        {
            return this.Entries().GetEnumerator();
        }

        /// <summary>
        /// Gets the number of items in the set.
        /// </summary>
        /// <returns>The number of elements in the set.</returns>
        public int Size()
        {
            this.slimLock.EnterReadLock();
            try
            {
                return this.set.Count;
            }
            finally
            {
                this.slimLock.ExitReadLock();
            }
        }

        /// <inheritdoc/>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}
