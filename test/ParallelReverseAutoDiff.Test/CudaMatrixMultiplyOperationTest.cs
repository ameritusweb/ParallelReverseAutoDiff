using ManagedCuda.VectorTypes;
using ParallelReverseAutoDiff.RMAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class CudaMatrixMultiplyOperationTest
    {
        [Fact]
        public async Task Forward_GivenTwoMatricesCreatedOnTheMainThread_MultipliesThemOnAnotherThreadAndTheResultIsNotNull()
        {
            CudaBlas.Instance.Initialize();
            await Task.Delay(5000);
            try
            {
                CudaMatrixMultiplyOperation op = new CudaMatrixMultiplyOperation();
                Matrix? c = null;
                Matrix a = new Matrix(100, 100);
                Matrix b = new Matrix(100, 1);
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