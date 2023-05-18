using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Convolutional;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class ConvolutionalNeuralNetworkTest
    {
        [Fact]
        public async Task GivenConvolutionalNeuralNetwork_UsesCudaOperationsSuccessfully()
        {
            CudaBlas.Instance.Initialize();
            try
            {
                ConvolutionalNeuralNetwork neuralNetwork = new ConvolutionalNeuralNetwork(
                    new Dimension { Depth = 12, Height = 8, Width = 8 },
                    new Dimension { Depth = 12, Height = 2, Width = 2 },
                    30976,
                    10000,
                    2048,
                    32,
                    4,
                    0.001d, 
                    null
                );
                await neuralNetwork.Initialize();
                for (int i = 0; i < 2; i++)
                {
                    DeepMatrix input = new DeepMatrix(12, 8, 8);
                    Matrix target = new Matrix(2048, 1);
                    target[0][0] = 0.5d;
                    await neuralNetwork.Optimize(input, target, i, null);
                }
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}