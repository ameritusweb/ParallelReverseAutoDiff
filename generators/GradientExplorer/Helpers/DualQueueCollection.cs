using System;
using System.Collections.Concurrent;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace GradientExplorer.Helpers
{
    public class DualQueueCollection<T> : IQueue<T>, IEnumerable
    {
        private readonly ConcurrentQueue<T> source = new ConcurrentQueue<T>();
        private readonly ConcurrentQueue<T> destination = new ConcurrentQueue<T>();
        private IEnumerator<T> initialFillEnumerator;

        public event EventHandler SourceEmptied;
        public event EventHandler<T> ItemProcessed;

        public Func<IEnumerable<T>> InitialFill { get; set; }

        public DualQueueCollection()
        {
        }

        public IReadOnlyList<T> Source => source.ToList().AsReadOnly();
        public IReadOnlyList<T> Destination => destination.ToList().AsReadOnly();

        public void Enqueue(T item)
        {
            source.Enqueue(item);
        }

        public bool TryDequeue(out T item)
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

        public bool TryPeek(out T item)
        {
            return source.TryPeek(out item);
        }

        public void Clear()
        {
            while (source.TryDequeue(out _)) ;
        }

        public int Count => source.Count;  // Be cautious about performance

        public bool IsEmpty
        {
            get
            {
                if (!source.IsEmpty)
                {
                    return false;
                }

                if (initialFillEnumerator != null)
                {
                    return !initialFillEnumerator.MoveNext();
                }

                return true;
            }
        }

        public void CopyTo(T[] array, int index)
        {
            ((ICollection)this).CopyTo(array, index);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return source.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public T[] ToArray()
        {
            return source.ToArray();
        }

        private void LazyInitializeSource()
        {
            lock (source)  // Ensuring thread safety
            {
                if (source.IsEmpty && destination.IsEmpty)
                {
                    if (InitialFill != null)
                    {
                        initialFillEnumerator = InitialFill().GetEnumerator();
                        while (initialFillEnumerator.MoveNext())
                        {
                            source.Enqueue(initialFillEnumerator.Current);
                        }
                    }
                }
            }
        }
    }
}
