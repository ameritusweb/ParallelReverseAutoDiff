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
                MatrixMultiplyOperation mmop = new MatrixMultiplyOperation();
                Matrix? c = null;
                Matrix? d = null;
                Matrix? f = null;
                Matrix? g = null;
                Matrix? h = null;
                Matrix? i = null;
                Matrix a = new Matrix(1000, 1000);
                a.Initialize(InitializationType.Xavier);
                Matrix b = new Matrix(1000, 1);
                b.Initialize(InitializationType.Xavier);
                Matrix gradient = new Matrix(1000, 1);
                gradient.Initialize(InitializationType.Xavier);
                int precision = 10;
                await Task.Run(() =>
                {
                    c = op.Forward(a, b);
                    d = mmop.Forward(a, b);
                    for (int j = 0; j < 1000; ++j)
                    {
                        Assert.Equal(Math.Round(c[j][0], precision), Math.Round(d[j][0], precision));
                    }
                    f = op.Backward(gradient).Item1 as Matrix;
                    g = op.Backward(gradient).Item2 as Matrix;
                    h = mmop.Backward(gradient).Item1 as Matrix;
                    i = mmop.Backward(gradient).Item2 as Matrix;
                    for (int j = 0; j < 1000; ++j)
                    {
                        for (int k = 0; k < 1000; ++k)
                        {
                            Assert.Equal(Math.Round(f[j][k], precision), Math.Round(h[j][k], precision));
                        }
                    }
                    for (int j = 0; j < 1000; ++j)
                    {
                        Assert.Equal(Math.Round(g[j][0], precision), Math.Round(i[j][0], precision));
                    }
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