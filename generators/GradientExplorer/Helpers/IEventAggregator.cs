using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
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
        void Publish<T>(EventType eventType, T eventData) where T : IEventData;

        /// <summary>
        /// Posts a message of a specific type to the event aggregator.
        /// </summary>
        /// <typeparam name="T">The type of the message.</typeparam>
        /// <param name="messageType">The message type enumeration value.</param>
        /// <param name="message">The message to post.</param>
        void PostMessage<T>(MessageType messageType, T message);

        /// <summary>
        /// Tries to retrieve a message of a specific type from the event aggregator.
        /// </summary>
        /// <typeparam name="T">The expected type of the message.</typeparam>
        /// <param name="messageType">The message type enumeration value.</param>
        /// <param name="message">The retrieved message, or the default value of T if not found.</param>
        /// <returns>True if a message of the specified type was found, otherwise false.</returns>
        bool TryRetrieveMessage<T>(MessageType messageType, out T message);

        /// <summary>
        /// Removes a message of a specific type from the event aggregator.
        /// </summary>
        /// <param name="messageType">The message type enumeration value.</param>
        /// <returns>True if the message was successfully removed, otherwise false.</returns>
        bool RemoveMessage(MessageType messageType);
    }
}
