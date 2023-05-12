using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.FeedForward;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class FeedForwardNeuralNetworkTest
    {
        [Fact]
        public async Task GivenFeedForwardNetwork_UsesCudaOperationsSuccessfully()
        {
            CudaBlas.Instance.Initialize();
            await Task.Delay(5000);
            try
            {
                FeedForwardNeuralNetwork neuralNetwork = new FeedForwardNeuralNetwork(100, 1000, 1, 3, 0.001d, null);
                await neuralNetwork.Initialize();
                for (int i = 0; i < 100; i++)
                {
                    Matrix input = new Matrix(100, 1);
                    Matrix target = new Matrix(1, 1);
                    target[0][0] = 0.5d;
                    try
                    {
                        await neuralNetwork.Optimize(input, target, i, null);
                    }
                    finally
                    {
                        CudaBlas.Instance.Dispose();
                    }
                }
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}