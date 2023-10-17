using System;
using System.Collections.Concurrent;

namespace GradientExplorer.Helpers
{
    public static class ConcurrentQueueExtensions
    {
        // Extension method to transfer items from another queue into the current queue
        public static void TransferIntoSelfFrom<T>(this ConcurrentQueue<T> destinationQueue, ConcurrentQueue<T> sourceQueue)
        {
            // Validate the input queue
            if (sourceQueue == null)
            {
                throw new ArgumentNullException(nameof(sourceQueue), "Source queue cannot be null.");
            }

            T item;
            while (sourceQueue.TryDequeue(out item))
            {
                destinationQueue.Enqueue(item);
            }
        }
    }
}
