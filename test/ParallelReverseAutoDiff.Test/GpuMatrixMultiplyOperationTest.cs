using ManagedCuda.VectorTypes;
using ParallelReverseAutoDiff.RMAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class GpuMatrixMultiplyOperationTest
    {
        [Fact]
        public async Task Forward_GivenTwoMatricesCreatedOnTheMainThread_MultipliesThemOnAnotherThreadAndTheResultIsNotNull()
        {
            CudaBlas.Instance.Initialize();
            await Task.Delay(5000);
            try
            {
                GpuMatrixMultiplyOperation op = new GpuMatrixMultiplyOperation();
                Matrix? c = null;
                DeepMatrix? f = null;
                Matrix a = new Matrix(1000, 1000);
                a.Initialize(InitializationType.Xavier);
                Matrix b = new Matrix(1000, 1);
                b.Initialize(InitializationType.Xavier);
                await Task.Run(() =>
                {
                    c = op.Forward(a, b);
                });
                Assert.NotNull(c);
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}