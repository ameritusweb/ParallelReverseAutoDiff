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
                GraphAttentionPathsNeuralNetwork neuralNetwork = new GraphAttentionPathsNeuralNetwork(10, 2, 4, 0.001d, 4d);
                neuralNetwork.Forward();
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}