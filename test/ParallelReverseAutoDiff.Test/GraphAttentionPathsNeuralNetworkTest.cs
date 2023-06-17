using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.GraphAttentionPaths;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class GraphAttentionPathsNeuralNetworkTest
    {
        [Fact]
        public async Task GivenGraphAttentionPathsNeuralNetwork_UsesCudaOperationsSuccessfully()
        {
            CudaBlas.Instance.Initialize();
            try
            {
                GraphAttentionPathsNeuralNetwork neuralNetwork = new GraphAttentionPathsNeuralNetwork();
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}