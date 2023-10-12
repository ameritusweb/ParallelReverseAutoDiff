using GradientExplorer.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface IMessagePoster
    {
        /// <summary>
        /// Posts a message of a certain type.
        /// </summary>
        /// <typeparam name="T">Type of the message.</typeparam>
        /// <param name="messageType">Type of message.</param>
        /// <param name="message">Message object.</param>
        void PostMessage<T>(MessageType messageType, T message);

        /// <summary>
        /// Removes a message of a certain type.
        /// </summary>
        /// <param name="messageType">Type of message.</param>
        /// <returns>True if the message is successfully removed, false otherwise.</returns>
        bool RemoveMessage(MessageType messageType);
    }
}
