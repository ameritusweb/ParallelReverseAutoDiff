using ParallelReverseAutoDiff.PRAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class TensorTests
    {
        [Fact]
        public void TestSlice3DTensors()
        {
            // Arrange
            Tensor[] tensors = {
                new Tensor(new int[] { 6, 6, 6 }, Enumerable.Range(0, 216).Select(i => (double)i).ToArray())
            };
            int[] sliceSizes = { 2, 2, 2 };

            // Act
            List<Tensor> slices = Tensor.Slice3DTensors(tensors, sliceSizes);

            // Assert
            Assert.Equal(27, slices.Count); // There should be 27 slices for a 6x6x6 tensor with 2x2x2 slice sizes

            // Verify the first slice
            Tensor firstSlice = slices.First();
            double[] expectedFirstSliceData = {
                0, 1,
                6, 7,
                36, 37,
                42, 43
            };
            Assert.Equal(new int[] { 2, 2, 2 }, firstSlice.Shape);
            Assert.Equal(expectedFirstSliceData, firstSlice.Data);

            // Print all slice data for debugging
            for (int i = 0; i < slices.Count; i++)
            {
                Console.WriteLine($"Slice {i + 1} Data: [{string.Join(", ", slices[i].Data)}]");
            }

            // Verify the last slice
            Tensor lastSlice = slices.Last();
            double[] expectedLastSliceData = {
                172, 173,
                178, 179,
                208, 209,
                214, 215
            };
            Assert.Equal(new int[] { 2, 2, 2 }, lastSlice.Shape);
            Assert.Equal(expectedLastSliceData, lastSlice.Data);
        }

        [Fact]
        public void Transpose_1_2_3_Shape_With_0_2_1_Permutation_ShouldReturnCorrectResult()
        {
            // Arrange
            double[] data = { 1, 2, 3, 4, 5, 6 };
            int[] shape = { 1, 2, 3 };
            Tensor tensor = new Tensor(shape, data);

            int[] permutation = { 0, 2, 1 };

            // Act
            Tensor result = tensor.Transpose(permutation);

            // Assert
            Assert.Equal(new int[] { 1, 3, 2 }, result.Shape);

            double[,,] expectedData = new double[1, 3, 2]
            {
            {
                { 1, 4 },
                { 2, 5 },
                { 3, 6 }
            }
            };

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        Assert.Equal(expectedData[i, j, k], result.Data[i * 6 + j * 2 + k]);
                    }
                }
            }
        }
    }
}
