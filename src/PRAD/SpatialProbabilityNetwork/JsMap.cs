//------------------------------------------------------------------------------
// <copyright file="JsMap.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.SpatialProbabilityNetwork
{
    using System;
    using System.Collections.Generic;
    using System.Threading;

    /// <summary>
    /// Thread-safe JavaScript-like Map implementation using a Dictionary.
    /// </summary>
    public class JsMap : IEnumerable<Tuple<object, object>>
    {
        private readonly Dictionary<object, object> map = new Dictionary<object, object>();
        private readonly ReaderWriterLockSlim slimLock = new ReaderWriterLockSlim();

        /// <summary>
        /// Sets a key-value pair in the map. If the key already exists, its value will be updated.
        /// </summary>
        /// <param name="key">The key of the entry.</param>
        /// <param name="value">The value associated with the key.</param>
        public void Set(object key, object value)
        {
            this.slimLock.EnterWriteLock();
            try
            {
                this.map[key] = value;
            }
            finally
            {
                this.slimLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Gets the value associated with the specified key.
        /// </summary>
        /// <param name="key">The key to look up.</param>
        /// <returns>The value associated with the key, or <c>null</c> if the key is not found.</returns>
        public object Get(object key)
        {
            this.slimLock.EnterReadLock();
            try
            {
                return this.map.TryGetValue(key, out var value) ? value : null!;
            }
            finally
            {
                this.slimLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Checks if the map contains the specified key.
        /// </summary>
        /// <param name="key">The key to check for existence.</param>
        /// <returns><c>true</c> if the key exists in the map; otherwise, <c>false</c>.</returns>
        public bool Has(object key)
        {
            this.slimLock.EnterReadLock();
            try
            {
                return this.map.ContainsKey(key);
            }
            finally
            {
                this.slimLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Deletes a key-value pair from the map by its key.
        /// </summary>
        /// <param name="key">The key of the entry to delete.</param>
        /// <returns><c>true</c> if the entry was successfully deleted; otherwise, <c>false</c>.</returns>
        public bool Delete(object key)
        {
            this.slimLock.EnterWriteLock();
            try
            {
                return this.map.Remove(key);
            }
            finally
            {
                this.slimLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Retrieves all key-value pairs in the map as tuples.
        /// </summary>
        /// <returns>An enumerable collection of tuples containing the key-value pairs.</returns>
        public IEnumerable<Tuple<object, object>> Entries()
        {
            this.slimLock.EnterReadLock();
            try
            {
                foreach (var kvp in this.map)
                {
                    yield return new Tuple<object, object>(kvp.Key, kvp.Value);
                }
            }
            finally
            {
                this.slimLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Gets an enumerator for iterating through the key-value pairs in the map.
        /// </summary>
        /// <returns>An enumerator for the map.</returns>
        public IEnumerator<Tuple<object, object>> GetEnumerator()
        {
            return this.Entries().GetEnumerator();
        }

        /// <summary>
        /// Clears all entries from the map.
        /// </summary>
        public void Clear()
        {
            this.slimLock.EnterWriteLock();
            try
            {
                this.map.Clear();
            }
            finally
            {
                this.slimLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Gets the number of key-value pairs currently in the map.
        /// </summary>
        /// <returns>The number of entries in the map.</returns>
        public int Size()
        {
            this.slimLock.EnterReadLock();
            try
            {
                return this.map.Count;
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
