//------------------------------------------------------------------------------
// <copyright file="ConcurrentHashSet.cs" author="ameritusweb" date="5/9/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Interprocess
{
    using System.Collections;
    using System.Collections.Concurrent;
    using System.Collections.Generic;

    /// <summary>
    /// A concurrent hash set.
    /// </summary>
    /// <typeparam name="T">The type of hash set.</typeparam>
    public class ConcurrentHashSet<T> : ICollection<T>, IEnumerable<T>, IEnumerable
    {
        private ConcurrentDictionary<T, byte> internalDictionary = new ConcurrentDictionary<T, byte>();

        /// <summary>
        /// Gets the count.
        /// </summary>
        public int Count
        {
            get { return this.internalDictionary.Count; }
        }

        /// <summary>
        /// Gets a value indicating whether the hash set is read only.
        /// </summary>
        public bool IsReadOnly
        {
            get { return false; }
        }

        /// <summary>
        /// Add an item.
        /// </summary>
        /// <param name="item">The item.</param>
        /// <returns>A success boolean.</returns>
        public bool Add(T item)
        {
            return this.internalDictionary.TryAdd(item, default(byte));
        }

        /// <summary>
        /// Except with.
        /// </summary>
        /// <param name="other">The other enumerable.</param>
        public void ExceptWith(IEnumerable<T> other)
        {
            foreach (var item in other)
            {
                byte outByte;
                this.internalDictionary.TryRemove(item, out outByte);
            }
        }

        /// <summary>
        /// Clears the hash set.
        /// </summary>
        public void Clear()
        {
            this.internalDictionary.Clear();
        }

        /// <summary>
        /// Contains an item.
        /// </summary>
        /// <param name="item">The item.</param>
        /// <returns>A success boolean.</returns>
        public bool Contains(T item)
        {
            return this.internalDictionary.ContainsKey(item);
        }

        /// <summary>
        /// Copy to an array.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="arrayIndex">The array index.</param>
        public void CopyTo(T[] array, int arrayIndex)
        {
            this.internalDictionary.Keys.CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// Remove an item.
        /// </summary>
        /// <param name="item">The item.</param>
        /// <returns>A success boolean.</returns>
        public bool Remove(T item)
        {
            byte outByte;
            return this.internalDictionary.TryRemove(item, out outByte);
        }

        /// <summary>
        /// Add an item.
        /// </summary>
        /// <param name="item">The item.</param>
        void ICollection<T>.Add(T item)
        {
            this.Add(item);
        }

        /// <summary>
        /// Gets an enumerator.
        /// </summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            return this.internalDictionary.Keys.GetEnumerator();
        }

        /// <summary>
        /// Get an enumerator.
        /// </summary>
        /// <returns>The enumerator.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.internalDictionary.Keys.GetEnumerator();
        }
    }
}
