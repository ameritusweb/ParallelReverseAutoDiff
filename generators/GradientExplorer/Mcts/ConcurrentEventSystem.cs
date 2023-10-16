using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public class ConcurrentEventSystem
    {
        // Thread-safe queue to hold events
        private ConcurrentQueue<Action> eventQueue;

        public ConcurrentEventSystem()
        {
            eventQueue = new ConcurrentQueue<Action>();
        }

        // Method to add an event to the queue
        public void EnqueueEvent(Action eventAction)
        {
            eventQueue.Enqueue(eventAction);
        }

        // Method to trigger events based on certain conditions
        public void TriggerEvents()
        {
            // Implement logic to trigger events, if necessary
        }

        // Background task to listen and process events
        public async Task EventListener()
        {
            while (true) // Add exit conditions as necessary
            {
                if (!eventQueue.IsEmpty)
                {
                    if (eventQueue.TryDequeue(out Action eventAction))
                    {
                        eventAction();
                    }
                }

                await Task.Delay(10); // To prevent tight-loop; adjust as needed
            }
        }
    }
}
