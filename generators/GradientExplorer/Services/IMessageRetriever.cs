using GradientExplorer.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface IMessageRetriever
    {
        /// <summary>
        /// Retrieves a message of a certain type.
        /// </summary>
        /// <typeparam name="T">Type of the message.</typeparam>
        /// <param name="messageType">Type of message.</param>
        /// <returns>Retrieved message object.</returns>
        T RetrieveMessage<T>(MessageType messageType);
    }
}
