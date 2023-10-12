using GradientExplorer.Helpers;
using System.Threading;

namespace GradientExplorer.Services
{
    public interface IEventAggregator
    {
        /// <summary>
        /// Subscribes a synchronous action to an event type with a given priority.
        /// </summary>
        /// <typeparam name="T">The event data type.</typeparam>
        /// <param name="eventType">The event type to subscribe to.</param>
        /// <param name="action">The action to execute when the event is published.</param>
        /// <param name="priority">The priority level for this subscriber.</param>
        /// <param name="filter">An optional filter function to further refine event handling.</param>
        /// <returns>A subscription object that can be disposed to unsubscribe.</returns>
        Subscription<T> Subscribe<T>(EventType eventType, Action<T, CancellationToken> action, int priority, Func<T, bool> filter = null) where T : IEventData;

        /// <summary>
        /// Subscribes an asynchronous action to an event type with a given priority.
        /// </summary>
        /// <typeparam name="T">The event data type.</typeparam>
        /// <param name="eventType">The event type to subscribe to.</param>
        /// <param name="asyncAction">The asynchronous action to execute when the event is published.</param>
        /// <param name="priority">The priority level for this subscriber.</param>
        /// <param name="filter">An optional filter function to further refine event handling.</param>
        /// <returns>A subscription object that can be disposed to unsubscribe.</returns>
        SubscriptionAsync<T> SubscribeAsync<T>(EventType eventType, Func<T, CancellationToken, Task> asyncAction, int priority, Func<T, bool> filter = null) where T : IEventData;

        /// <summary>
        /// Publishes an event of a given type.
        /// </summary>
        /// <typeparam name="T">The event data type.</typeparam>
        /// <param name="eventType">The event type to publish.</param>
        /// <param name="eventData">The event data to publish.</param>
        Task PublishAsync<T>(EventType eventType, T eventData) where T : IEventData;
    }
}
