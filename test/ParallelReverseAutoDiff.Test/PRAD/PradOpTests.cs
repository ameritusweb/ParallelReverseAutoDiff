using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.Test.Common;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class PradOpTests
    {
        [Fact]
        public void TestInitialization()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var pradOp = new PradOp(seed);

            Assert.NotNull(pradOp);
        }

        [Fact]
        public void TestElementwiseAddition()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var tensorToAdd = new Tensor(new int[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Add(tensorToAdd);

            Assert.Equal(new double[] { 6, 8, 10, 12 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestElementwiseSubtraction()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
            var tensorToSub = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Sub(tensorToSub);

            Assert.Equal(new double[] { 4, 4, 4, 4 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestElementwiseMultiplication()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var tensorToMul = new Tensor(new int[] { 2, 2 }, new double[] { 2, 2, 2, 2 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Mul(tensorToMul);

            Assert.Equal(new double[] { 2, 4, 6, 8 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 2, 2, 2, 2 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestElementwiseSin()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 0, Math.PI / 2, Math.PI, 3 * Math.PI / 2 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Sin();

            Assert.Equal(new double[] { 0, 1, 0, -1 }, result.Result.Data, new DoubleArrayEqualityComparer(5d));

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 0, -1, 0 }, result.Gradients[0].Data, new DoubleArrayEqualityComparer(5d));
        }

        [Fact]
        public void TestElementwiseCos()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 0, Math.PI / 2, Math.PI, 3 * Math.PI / 2 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Cos();

            Assert.Equal(new double[] { 1, 0, -1, 0 }, result.Result.Data, new DoubleArrayEqualityComparer(5d));

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 0, -1, 0, 1 }, result.Gradients[0].Data, new DoubleArrayEqualityComparer(5d));
        }

        [Fact]
        public void TestReshape()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Reshape(new int[] { 4, 1 });

            Assert.Equal(new double[] { 1, 2, 3, 4 }, result.Result.Data);
            Assert.Equal(new int[] { 4, 1 }, result.Result.Shape);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 4, 1 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestTile()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Tile(new int[] { 2, 2 });

            Assert.Equal(new double[] { 1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 4, 4 }, new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 4, 4, 4, 4 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestGather()
        {
            var seed = new Tensor(new int[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var indices = new Tensor(new int[] { 2 }, new double[] { 2, 0 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Gather(indices, axis: 0);

            Assert.Equal(new double[] { 5, 6, 1, 2 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 1, 0, 0, 1, 1 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestSlice()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Slice(new int[] { 0, 0 }, new int[] { 2, 1 });

            Assert.Equal(new double[] { 1, 3 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 1 }, new double[] { 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 0, 1, 0 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestGatherNd()
        {
            var seed = new Tensor(new int[] { 3, 3 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var indices = new Tensor(new int[] { 2, 2 }, new double[] { 0, 1, 2, 0 });
            var pradOp = new PradOp(seed);

            var result = pradOp.GatherNd(indices);

            Assert.Equal(new double[] { 2, 7 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2 }, new double[] { 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 0, 1, 0, 0, 0, 0, 1, 0, 0 }, result.Gradients[0].Data);
        }
    }
}
