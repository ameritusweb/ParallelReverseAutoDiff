namespace GradientExplorer.Helpers
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading;

    public class ThreadSafeList<T> : IList<T>, IDisposable
    {
        private readonly List<T> _innerList = new List<T>();
        private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();

        public T this[int index]
        {
            get => ReadOperation(innerList => innerList[index]);
            set => WriteOperation(innerList => innerList[index] = value);
        }

        public int Count => ReadOperation(innerList => innerList.Count);

        public bool IsReadOnly => false;

        public void Add(T item) => WriteOperation(innerList => innerList.Add(item));

        public void Clear() => WriteOperation(innerList => innerList.Clear());

        public bool Contains(T item) => ReadOperation(innerList => innerList.Contains(item));

        public void CopyTo(T[] array, int arrayIndex) => ReadOperation(innerList => innerList.CopyTo(array, arrayIndex));

        public IEnumerator<T> GetEnumerator() => ReadOperation(innerList => new List<T>(innerList).GetEnumerator());

        public int IndexOf(T item) => ReadOperation(innerList => innerList.IndexOf(item));

        public void Insert(int index, T item) => WriteOperation(innerList => innerList.Insert(index, item));

        public bool Remove(T item) => WriteOperation(innerList => innerList.Remove(item));

        public void RemoveAt(int index) => WriteOperation(innerList => innerList.RemoveAt(index));

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public void Dispose() => _lock.Dispose();

        private void WriteOperation(Action<List<T>> operation)
        {
            _lock.EnterWriteLock();
            try
            {
                operation(_innerList);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        // New WriteOperation method for operations that return a value
        private TResult WriteOperation<TResult>(Func<List<T>, TResult> operation)
        {
            _lock.EnterWriteLock();
            try
            {
                return operation(_innerList);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        private TResult ReadOperation<TResult>(Func<List<T>, TResult> operation)
        {
            _lock.EnterReadLock();
            try
            {
                return operation(_innerList);
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        // New ReadOperation method for void-returning operations
        private void ReadOperation(Action<List<T>> operation)
        {
            _lock.EnterReadLock();
            try
            {
                operation(_innerList);
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }

}
