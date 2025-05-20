using ParallelReverseAutoDiff.PRAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class ElementwiseMultiplyBroadcastingReverseTests
    {
        [Fact]
        public void SameShape_ReturnsCorrectResult()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var b = new Tensor(new[] { 2, 2 }, new double[] { 5, 6, 7, 8 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(new double[] { 5, 12, 21, 32 }, result.Data);
            Assert.Equal(new[] { 2, 2 }, result.Shape);
            Assert.Equal(new[] { 0, 1, 2, 3 }, mapping.SourceIndicesA);
            Assert.Equal(new[] { 0, 1, 2, 3 }, mapping.SourceIndicesB);
        }

        [Fact]
        public void ScalarBroadcast_ReturnsCorrectResult()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var b = new Tensor(new[] { 1, 1 }, new double[] { 2 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(new double[] { 2, 4, 6, 8 }, result.Data);
            Assert.Equal(new[] { 2, 2 }, result.Shape);
            Assert.Equal(new[] { 0, 1, 2, 3 }, mapping.SourceIndicesA);
            Assert.Equal(new[] { 0, 0, 0, 0 }, mapping.SourceIndicesB);
        }

        [Fact]
        public void RowBroadcast_ReturnsCorrectResult()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var b = new Tensor(new[] { 1, 3 }, new double[] { 7, 8, 9 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(new double[] { 7, 16, 27, 28, 40, 54 }, result.Data);
            Assert.Equal(new[] { 2, 3 }, result.Shape);
            Assert.Equal(new[] { 0, 1, 2, 3, 4, 5 }, mapping.SourceIndicesA);
            Assert.Equal(new[] { 0, 1, 2, 0, 1, 2 }, mapping.SourceIndicesB);
        }

        [Fact]
        public void ColumnBroadcast_ReturnsCorrectResult()
        {
            // Arrange
            var a = new Tensor(new[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var b = new Tensor(new[] { 3, 1 }, new double[] { 7, 8, 9 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(new double[] { 7, 14, 24, 32, 45, 54 }, result.Data);
            Assert.Equal(new[] { 3, 2 }, result.Shape);
            Assert.Equal(new[] { 0, 1, 2, 3, 4, 5 }, mapping.SourceIndicesA);
            Assert.Equal(new[] { 0, 0, 1, 1, 2, 2 }, mapping.SourceIndicesB);
        }

        [Fact]
        public void ThreeDimensionalBroadcast_ReturnsCorrectResult()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2, 2 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var b = new Tensor(new[] { 1, 2, 2 }, new double[] { 9, 10, 11, 12 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(new double[] { 9, 20, 33, 48, 45, 60, 77, 96 }, result.Data);
            Assert.Equal(new[] { 2, 2, 2 }, result.Shape);
            Assert.Equal(new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, mapping.SourceIndicesA);
            Assert.Equal(new[] { 0, 1, 2, 3, 0, 1, 2, 3 }, mapping.SourceIndicesB);
        }

        [Fact]
        public void LargeArrays_ReturnsCorrectResult()
        {
            // Arrange
            int size = 1000;
            var aData = Enumerable.Range(1, size).Select(x => (double)x).ToArray();
            var bData = Enumerable.Range(1, size).Select(x => (double)x).ToArray();
            var a = new Tensor(new[] { size }, aData);
            var b = new Tensor(new[] { size }, bData);

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            var expected = aData.Zip(bData, (x, y) => x * y).ToArray();
            Assert.Equal(expected, result.Data);
            Assert.Equal(new[] { size }, result.Shape);
            Assert.Equal(Enumerable.Range(0, size).ToArray(), mapping.SourceIndicesA);
            Assert.Equal(Enumerable.Range(0, size).ToArray(), mapping.SourceIndicesB);
        }

        [Fact]
        public void IncompatibleShapes_ThrowsArgumentException()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var b = new Tensor(new[] { 2, 2 }, new double[] { 7, 8, 9, 10 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => a.ElementwiseMultiplyBroadcasting(b));
        }

        [Fact]
        public void ZeroDimension_ReturnsEmptyTensor()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 0, 2 }, Array.Empty<double>());
            var b = new Tensor(new[] { 2, 0, 1 }, Array.Empty<double>());

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Empty(result.Data);
            Assert.Equal(new[] { 2, 0, 2 }, result.Shape);
            Assert.Empty(mapping.SourceIndicesA);
            Assert.Empty(mapping.SourceIndicesB);
        }

        [Fact]
        public void BroadcastMapping_PreservesCorrectIndices()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var b = new Tensor(new[] { 1, 2 }, new double[] { 5, 6 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(4, mapping.SourceIndicesA.Length);
            Assert.Equal(4, mapping.SourceIndicesB.Length);
            Assert.Equal(new[] { 2, 2 }, mapping.ResultShape);

            // Verify the mapping correctly maps to original indices
            for (int i = 0; i < result.Data.Length; i++)
            {
                Assert.Equal(
                    result.Data[i],
                    a.Data[mapping.SourceIndicesA[i]] * b.Data[mapping.SourceIndicesB[i]]);
            }
        }

        [Fact]
        public void SIMD_HandlesUnalignedData()
        {
            // Arrange
            var a = new Tensor(new[] { 5 }, new double[] { 1, 2, 3, 4, 5 });  // Unaligned length
            var b = new Tensor(new[] { 1 }, new double[] { 2 });

            // Act
            var (result, mapping) = a.ElementwiseMultiplyBroadcasting(b)!.Value;

            // Assert
            Assert.Equal(new double[] { 2, 4, 6, 8, 10 }, result.Data);
        }

        [Fact]
        public void SameShape_ComputesCorrectGradients()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var b = new Tensor(new[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
            var upstream = new Tensor(new[] { 2, 2 }, new double[] { 1, 1, 1, 1 });

            // Simple mapping for same shapes
            var mapping = new BroadcastMapping(
                Enumerable.Range(0, 4).ToArray(),
                Enumerable.Range(0, 4).ToArray(),
                new[] { 2, 2 },
                Enumerable.Range(0, 4).ToArray(),
                Enumerable.Range(0, 4).ToArray());

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Equal(2, gradients.Length);
            Assert.Equal(new double[] { 5, 6, 7, 8 }, gradients[0].Data); // dL/dA = upstream * B
            Assert.Equal(new double[] { 1, 2, 3, 4 }, gradients[1].Data); // dL/dB = upstream * A
        }

        [Fact]
        public void ScalarBroadcast_ComputesCorrectGradients()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var b = new Tensor(new[] { 1, 1 }, new double[] { 2 });
            var upstream = new Tensor(new[] { 2, 2 }, new double[] { 1, 1, 1, 1 });

            // Create mapping for scalar broadcast
            var sourceIndicesA = new[] { 0, 1, 2, 3 };
            var sourceIndicesB = new[] { 0, 0, 0, 0 }; // Scalar broadcasts to all positions
            var mapping = new BroadcastMapping(
                sourceIndicesA,
                sourceIndicesB,
                new[] { 2, 2 },
                sourceIndicesA,
                new[] { 0 });

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Equal(new double[] { 2, 2, 2, 2 }, gradients[0].Data); // dL/dA = upstream * B
            Assert.Equal(new double[] { 10 }, gradients[1].Data); // dL/dB = sum(upstream * A)
        }

        [Fact]
        public void RowBroadcast_ComputesCorrectGradients()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var b = new Tensor(new[] { 1, 3 }, new double[] { 7, 8, 9 });
            var upstream = new Tensor(new[] { 2, 3 }, new double[] { 1, 1, 1, 1, 1, 1 });

            // Create mapping for row broadcast
            var sourceIndicesA = new[] { 0, 1, 2, 3, 4, 5 };
            var sourceIndicesB = new[] { 0, 1, 2, 0, 1, 2 };
            var mapping = new BroadcastMapping(
                sourceIndicesA,
                sourceIndicesB,
                new[] { 2, 3 },
                sourceIndicesA,
                new[] { 0, 1, 2 });

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Equal(new double[] { 7, 8, 9, 7, 8, 9 }, gradients[0].Data); // dL/dA = upstream * B
            Assert.Equal(new double[] { 5, 7, 9 }, gradients[1].Data); // dL/dB = sum(upstream * A) for each column
        }

        [Fact]
        public void ColumnBroadcast_ComputesCorrectGradients()
        {
            // Arrange
            var a = new Tensor(new[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var b = new Tensor(new[] { 3, 1 }, new double[] { 7, 8, 9 });
            var upstream = new Tensor(new[] { 3, 2 }, new double[] { 1, 1, 1, 1, 1, 1 });

            // Create mapping for column broadcast
            var sourceIndicesA = new[] { 0, 1, 2, 3, 4, 5 };
            var sourceIndicesB = new[] { 0, 0, 1, 1, 2, 2 };
            var mapping = new BroadcastMapping(
                sourceIndicesA,
                sourceIndicesB,
                new[] { 3, 2 },
                sourceIndicesA,
                new[] { 0, 1, 2 });

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Equal(new double[] { 7, 7, 8, 8, 9, 9 }, gradients[0].Data); // dL/dA = upstream * B
            Assert.Equal(new double[] { 3, 7, 11 }, gradients[1].Data); // dL/dB = sum(upstream * A) for each row
        }

        [Fact]
        public void ThreeDimensionalBroadcast_ComputesCorrectGradients()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 2, 2 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var b = new Tensor(new[] { 1, 2, 2 }, new double[] { 9, 10, 11, 12 });
            var upstream = new Tensor(new[] { 2, 2, 2 }, new double[] { 1, 1, 1, 1, 1, 1, 1, 1 });

            // Create mapping for 3D broadcast
            var sourceIndicesA = Enumerable.Range(0, 8).ToArray();
            var sourceIndicesB = new[] { 0, 1, 2, 3, 0, 1, 2, 3 };
            var mapping = new BroadcastMapping(
                sourceIndicesA,
                sourceIndicesB,
                new[] { 2, 2, 2 },
                sourceIndicesA,
                new[] { 0, 1, 2, 3 });

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Equal(new double[] { 9, 10, 11, 12, 9, 10, 11, 12 }, gradients[0].Data);
            Assert.Equal(new double[] { 6, 8, 10, 12 }, gradients[1].Data);
        }

        [Fact]
        public void ZeroDimension_ReturnsZeroGradients()
        {
            // Arrange
            var a = new Tensor(new[] { 2, 0, 2 }, Array.Empty<double>());
            var b = new Tensor(new[] { 2, 0, 1 }, Array.Empty<double>());
            var upstream = new Tensor(new[] { 2, 0, 2 }, Array.Empty<double>());

            var mapping = new BroadcastMapping(
                Array.Empty<int>(),
                Array.Empty<int>(),
                new[] { 2, 0, 2 },
                Array.Empty<int>(),
                Array.Empty<int>());

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Empty(gradients[0].Data);
            Assert.Empty(gradients[1].Data);
            Assert.Equal(new[] { 2, 0, 2 }, gradients[0].Shape);
            Assert.Equal(new[] { 2, 0, 1 }, gradients[1].Shape);
        }

        [Fact]
        public void LargeArrays_ComputesCorrectGradients()
        {
            // Arrange
            int size = 1000;
            var aData = Enumerable.Range(1, size).Select(x => (double)x).ToArray();
            var bData = Enumerable.Range(1, size).Select(x => (double)x).ToArray();
            var upstreamData = Enumerable.Repeat(1d, size).ToArray();

            var a = new Tensor(new[] { size }, aData);
            var b = new Tensor(new[] { size }, bData);
            var upstream = new Tensor(new[] { size }, upstreamData);

            var mapping = new BroadcastMapping(
                Enumerable.Range(0, size).ToArray(),
                Enumerable.Range(0, size).ToArray(),
                new[] { size },
                Enumerable.Range(0, size).ToArray(),
                Enumerable.Range(0, size).ToArray());

            var reverseOp = new TensorReverse(new Tensor[] { a, b });

            // Act
            var gradients = reverseOp.ElementwiseMultiplyBroadcastingReverse(upstream, mapping);

            // Assert
            Assert.Equal(bData, gradients[0].Data);
            Assert.Equal(aData, gradients[1].Data);
        }

        [Fact]
        public void InvalidTensorCount_ThrowsInvalidOperationException()
        {
            // Arrange
            var reverseOp = new TensorReverse(new Tensor[] { new Tensor(new[] { 1 }, new double[] { 1 }) });

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                reverseOp.ElementwiseMultiplyBroadcastingReverse(
                    new Tensor(new[] { 1 }, new double[] { 1 }),
                    new BroadcastMapping(new[] { 0 }, new[] { 0 }, new[] { 1 }, new[] { 0 }, new[] { 0 })));
        }
    }
}
