using System;
using System.Collections.Generic;

namespace GradientExplorer.Helpers
{
    /// <summary>
    /// Generic interface for a queue data structure.
    /// </summary>
    /// <typeparam name="T">The type of elements in the queue.</typeparam>
    public interface IQueue<T>
    {
        /// <summary>
        /// Attempts to remove and return the object at the beginning of the queue.
        /// </summary>
        /// <param name="result">When this method returns, if the operation was successful, result contains the object removed. If no object was available to be removed, the value is unspecified.</param>
        /// <returns>true if an element was removed and returned from the queue; otherwise, false.</returns>
        bool TryDequeue(out T result);

        /// <summary>
        /// Adds an object to the end of the queue.
        /// </summary>
        /// <param name="item">The object to add to the queue. The value can be a null reference for reference types.</param>
        void Enqueue(T item);

        /// <summary>
        /// Gets whether the queue is empty or not.
        /// </summary>
        bool IsEmpty { get; }

        /// <summary>
        /// Attempts to return an object from the beginning of the queue without removing it.
        /// </summary>
        /// <param name="result">When this method returns, result contains an object from the queue or an unspecified value if the operation failed.</param>
        /// <returns>true if and object was returned successfully; otherwise, false.</returns>
        bool TryPeek(out T result);

        /// <summary>
        /// Copies the elements of the queue to an existing one-dimensional Array, starting at the specified array index.
        /// </summary>
        /// <param name="array">The one-dimensional Array that is the destination of the elements copied from the queue. The Array must have zero-based indexing.</param>
        /// <param name="arrayIndex">The zero-based index in array at which copying begins.</param>
        void CopyTo(T[] array, int arrayIndex);

        /// <summary>
        /// Removes all objects from the queue.
        /// </summary>
        void Clear();

        /// <summary>
        /// Gets the number of elements contained in the queue.
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Returns an enumerator that iterates through the queue.
        /// </summary>
        /// <returns>An enumerator for the queue.</returns>
        IEnumerator<T> GetEnumerator();
    }
}
