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

        [Fact]
        public void ExtractPatches_2DInput_ValidPadding_CorrectOutput()
        {
            // Arrange
            var input = new Tensor(new int[] { 4, 4 }, new double[]
            {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
            });

            int[] filterSize = { 2, 2 };
            int[] strides = { 1, 1 };
            string padding = "VALID";

            // Act
            var result = input.ExtractPatches(filterSize, strides, padding);

            // Assert
            Assert.Equal(new int[] { 1, 3, 3, 4 }, result.Shape);
            Assert.Equal(new double[]
            {
            1, 2, 5, 6,
            2, 3, 6, 7,
            3, 4, 7, 8,
            5, 6, 9, 10,
            6, 7, 10, 11,
            7, 8, 11, 12,
            9, 10, 13, 14,
            10, 11, 14, 15,
            11, 12, 15, 16
            }, result.Data);
        }

        [Fact]
        public void ExtractPatches_3DInput_SamePadding_CorrectOutput()
        {
            // Arrange
            var input = new Tensor(new int[] { 3, 3, 2 }, new double[]
            {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18
            });

            int[] filterSize = { 2, 2 };
            int[] strides = { 1, 1 };
            string padding = "SAME";

            // Act
            var result = input.ExtractPatches(filterSize, strides, padding);

            // Assert
            Assert.Equal(new int[] { 1, 3, 3, 8 }, result.Shape);
            // Check a few key values
            Assert.Equal(1, result.Data[0]);
            Assert.Equal(2, result.Data[1]);
            Assert.Equal(3, result.Data[2]);
            Assert.Equal(4, result.Data[3]);
            Assert.Equal(7, result.Data[4]);
            Assert.Equal(8, result.Data[5]);
            Assert.Equal(9, result.Data[6]);
            Assert.Equal(10, result.Data[7]);

            // Check the second patch (top-middle)
            Assert.Equal(3, result.Data[8]);
            Assert.Equal(4, result.Data[9]);
            Assert.Equal(5, result.Data[10]);
            Assert.Equal(6, result.Data[11]);
            Assert.Equal(9, result.Data[12]);
            Assert.Equal(10, result.Data[13]);
            Assert.Equal(11, result.Data[14]);
            Assert.Equal(12, result.Data[15]);

            // Check the third patch (top-right)
            Assert.Equal(5, result.Data[16]);
            Assert.Equal(6, result.Data[17]);
            Assert.Equal(0, result.Data[18]);  // Right padding
            Assert.Equal(0, result.Data[19]);  // Right padding
            Assert.Equal(11, result.Data[20]);
            Assert.Equal(12, result.Data[21]);
            Assert.Equal(0, result.Data[22]);  // Right padding
            Assert.Equal(0, result.Data[23]);  // Right padding

            // Check the last patch (bottom-right corner)
            int lastPatchStart = result.Data.Length - 8;
            Assert.Equal(17, result.Data[lastPatchStart]);
            Assert.Equal(18, result.Data[lastPatchStart + 1]);
            Assert.Equal(0, result.Data[lastPatchStart + 2]);  // Right padding
            Assert.Equal(0, result.Data[lastPatchStart + 3]);  // Right padding
            Assert.Equal(0, result.Data[lastPatchStart + 4]);  // Bottom padding
            Assert.Equal(0, result.Data[lastPatchStart + 5]);  // Bottom padding
            Assert.Equal(0, result.Data[lastPatchStart + 6]);  // Bottom-right padding
            Assert.Equal(0, result.Data[lastPatchStart + 7]);  // Bottom-right padding
        }

        [Fact]
        public void ExtractPatches_4DInput_ValidPadding_CorrectOutput()
        {
            // Arrange
            var input = new Tensor(new int[] { 2, 3, 3, 1 }, new double[]
            {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,

            10, 11, 12,
            13, 14, 15,
            16, 17, 18
            });

            int[] filterSize = { 2, 2 };
            int[] strides = { 1, 1 };
            string padding = "VALID";

            // Act
            var result = input.ExtractPatches(filterSize, strides, padding);

            // Assert
            Assert.Equal(new int[] { 2, 2, 2, 4 }, result.Shape);
            Assert.Equal(new double[]
            {
            1, 2, 4, 5,
            2, 3, 5, 6,
            4, 5, 7, 8,
            5, 6, 8, 9,

            10, 11, 13, 14,
            11, 12, 14, 15,
            13, 14, 16, 17,
            14, 15, 17, 18
            }, result.Data);
        }

        [Fact]
        public void ExtractPatches_InvalidFilterSize_ThrowsArgumentException()
        {
            // Arrange
            var input = new Tensor(new int[] { 4, 4 }, new double[16]);
            int[] filterSize = { 2 };
            int[] strides = { 1, 1 };
            string padding = "VALID";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => input.ExtractPatches(filterSize, strides, padding));
        }

        [Fact]
        public void ExtractPatches_InvalidStrides_ThrowsArgumentException()
        {
            // Arrange
            var input = new Tensor(new int[] { 4, 4 }, new double[16]);
            int[] filterSize = { 2, 2 };
            int[] strides = { 1 };
            string padding = "VALID";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => input.ExtractPatches(filterSize, strides, padding));
        }

        [Fact]
        public void ExtractPatches_InvalidPadding_ThrowsArgumentException()
        {
            // Arrange
            var input = new Tensor(new int[] { 4, 4 }, new double[16]);
            int[] filterSize = { 2, 2 };
            int[] strides = { 1, 1 };
            string padding = "INVALID";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => input.ExtractPatches(filterSize, strides, padding));
        }

        [Fact]
        public void ExtractPatches_1DInput_ThrowsArgumentException()
        {
            // Arrange
            var input = new Tensor(new int[] { 4 }, new double[4]);
            int[] filterSize = { 2, 2 };
            int[] strides = { 1, 1 };
            string padding = "VALID";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => input.ExtractPatches(filterSize, strides, padding));
        }
    }
}
