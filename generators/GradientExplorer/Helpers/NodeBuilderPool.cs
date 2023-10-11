namespace GradientExplorer.Services
{
    using GradientExplorer.Model;
    using System;
    using System.Collections.Concurrent;
    using System.Threading;
    using System.Threading.Tasks;

    public class NodeBuilderPool : INodeBuilderPool
    {
        private readonly ConcurrentDictionary<NodeType, ConcurrentQueue<NodeBuilder>> pool;
        private readonly int MaxPoolSize = 100; // Define a maximum pool size
        private const int InitialPoolSize = 10;
        private readonly TimeSpan WaitTime = TimeSpan.FromSeconds(5); // Time to wait before throwing exception

        public NodeBuilderPool()
        {
            pool = new ConcurrentDictionary<NodeType, ConcurrentQueue<NodeBuilder>>();
            InitializePool();
        }

        private void InitializePool()
        {
            foreach (NodeType type in Enum.GetValues(typeof(NodeType)))
            {
                pool[type] = new ConcurrentQueue<NodeBuilder>();
                Prepopulate(type);
            }
        }

        private void Prepopulate(NodeType type)
        {
            for (int i = 0; i < InitialPoolSize; i++)
            {
                var builder = new NodeBuilder(type);
                pool[type].Enqueue(builder);
            }
        }

        public async Task<NodeBuilder> GetNodeBuilderAsync(NodeType type)
        {
            NodeBuilder builder;
            var cancellationTokenSource = new CancellationTokenSource(WaitTime);

            while (!cancellationTokenSource.Token.IsCancellationRequested)
            {
                if (pool[type].TryDequeue(out builder))
                {
                    return builder;
                }

                // Limit the number of new instances
                if (pool[type].Count < MaxPoolSize)
                {
                    return new NodeBuilder(type);
                }

                await Task.Delay(100, cancellationTokenSource.Token); // Wait for 100 ms before next try
            }

            throw new TimeoutException("Could not serve a NodeBuilder: Pool limit reached.");
        }

        public void Reclaim(NodeBuilder builder)
        {
            builder.Reset();

            // Enqueue only if below max limit
            if (pool[builder.NodeType].Count < MaxPoolSize)
            {
                pool[builder.NodeType].Enqueue(builder);
            }
        }
    }

}
