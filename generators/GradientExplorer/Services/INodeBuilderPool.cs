namespace GradientExplorer.Services
{
    using GradientExplorer.Model;
    using System;
    using System.Threading.Tasks;

    public interface INodeBuilderPool
    {
        Task<NodeBuilder> GetNodeBuilderAsync(NodeType type);
        void Reclaim(NodeBuilder builder);
    }
}
