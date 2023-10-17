using System;
using System.Collections.Concurrent;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace GradientExplorer.Helpers
{
    public class DualQueueCollection<T> : IProducerConsumerCollection<T>
    {
        private readonly ConcurrentQueue<T> source = new ConcurrentQueue<T>();
        private readonly ConcurrentQueue<T> destination = new ConcurrentQueue<T>();

        public event EventHandler SourceEmptied;
        public event EventHandler<T> ItemProcessed;

        public Action<ConcurrentQueue<T>> InitialFill { get; set; }

        public DualQueueCollection()
        {
        }

        public IReadOnlyList<T> Source => source.ToList().AsReadOnly();
        public IReadOnlyList<T> Destination => destination.ToList().AsReadOnly();

        public bool TryAdd(T item)
        {
            source.Enqueue(item);
            return true;
        }

        public bool TryTake(out T item)
        {
            LazyInitializeSource();

            bool result = false;
            EventHandler sourceEmptied = null;

            lock (source)  // Ensuring atomicity
            {
                if (source.TryDequeue(out item))
                {
                    destination.Enqueue(item);
                    result = true;

                    if (source.IsEmpty)
                    {
                        sourceEmptied = SourceEmptied;
                    }
                }
            }

            if (result)
            {
                ItemProcessed?.Invoke(this, item);
            }

            sourceEmptied?.Invoke(this, EventArgs.Empty);

            return result;
        }

        public int Count => source.Count + destination.Count;  // Be cautious about performance.

        public bool IsSynchronized => false;

        public object SyncRoot => throw new NotSupportedException();

        public void CopyTo(T[] array, int index)
        {
            ((ICollection)this).CopyTo(array, index);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return source.Concat(destination).GetEnumerator();
        }

        void ICollection.CopyTo(Array array, int index)
        {
            int i = index;
            foreach (var item in this)
            {
                array.SetValue(item, i++);
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public T[] ToArray()
        {
            return source.Concat(destination).ToArray();
        }

        private void LazyInitializeSource()
        {
            lock (source)  // Added to ensure thread safety
            {
                if (source.IsEmpty && destination.IsEmpty)
                {
                    InitialFill?.Invoke(source);
                }
            }
        }
    }
}
