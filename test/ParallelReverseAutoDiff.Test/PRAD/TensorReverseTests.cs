using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.Test.Common;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class TensorReverseTests
    {
        [Fact]
        public void ElementwiseAddReverse_Test()
        {
            // Arrange
            var tensorA = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var tensorB = new Tensor(new int[] { 2, 2 }, new double[] { 5.0, 6.0, 7.0, 8.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });

            var tensorReverse = new TensorReverse(new Tensor[] { tensorA, tensorB });

            // Act
            var gradients = tensorReverse.ElementwiseAddReverse(upstreamGradient);

            // Assert
            Assert.Equal(upstreamGradient.Data, gradients[0].Data);
            Assert.Equal(upstreamGradient.Data, gradients[1].Data);
        }

        [Fact]
        public void ElementwiseMultiplyReverse_Test()
        {
            // Arrange
            var tensorA = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var tensorB = new Tensor(new int[] { 2, 2 }, new double[] { 5.0, 6.0, 7.0, 8.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });

            var tensorReverse = new TensorReverse(new Tensor[] { tensorA, tensorB });

            // Act
            var gradients = tensorReverse.ElementwiseMultiplyReverse(upstreamGradient);

            // Assert
            Assert.Equal(new double[] { 5.0, 6.0, 7.0, 8.0 }, gradients[0].Data);
            Assert.Equal(new double[] { 1.0, 2.0, 3.0, 4.0 }, gradients[1].Data);
        }

        [Fact]
        public void IndexerReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 3, 3 }, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var indices = new string[] { "0:2", "1:3" };

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.IndexerReverse(upstreamGradient, indices);

            // Assert
            var expectedGradient = new double[] { 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void ConcatReverse_Test()
        {
            // Arrange
            var tensorA = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var tensorB = new Tensor(new int[] { 2, 2 }, new double[] { 5.0, 6.0, 7.0, 8.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 4 }, new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var tensorReverse = new TensorReverse(new Tensor[] { tensorA, tensorB });

            // Act
            var gradients = tensorReverse.ConcatReverse(upstreamGradient, 1);

            // Assert
            Assert.Equal(new double[] { 1.0, 1.0, 1.0, 1.0 }, gradients[0].Data);
            Assert.Equal(new double[] { 1.0, 1.0, 1.0, 1.0 }, gradients[1].Data);
        }

        [Fact]
        public void StackReverse_Test()
        {
            // Arrange
            var tensorA = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var tensorB = new Tensor(new int[] { 2, 2 }, new double[] { 5.0, 6.0, 7.0, 8.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var tensorReverse = new TensorReverse(new Tensor[] { tensorA, tensorB });

            // Act
            var gradients = tensorReverse.StackReverse(upstreamGradient, 0);

            // Assert
            Assert.Equal(new double[] { 1.0, 1.0, 1.0, 1.0 }, gradients[0].Data);
            Assert.Equal(new double[] { 1.0, 1.0, 1.0, 1.0 }, gradients[1].Data);
        }

        [Fact]
        public void GatherReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 3, 3 }, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
            var indices = new Tensor(new int[] { 2 }, new double[] { 0, 2 });
            var upstreamGradient = new Tensor(new int[] { 2, 3 }, new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.GatherReverse(upstreamGradient, indices);

            // Assert
            var expectedGradient = new double[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void TileReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var upstreamGradient = new Tensor(new int[] { 4, 4 }, new double[]
            {
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                5.0, 6.0, 7.0, 8.0
            });

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.TileReverse(upstreamGradient, new int[] { 2, 2 });

            // Assert
            var expectedGradient = new double[] { 16.0, 20.0, 16.0, 20.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void TileReverse2_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 1, 4 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 4 }, new double[]
            {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            });

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.TileReverse(upstreamGradient, new int[] { 2, 1 });

            // Assert
            var expectedGradient = new double[] { 6.0, 8.0, 10.0, 12.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void TileReverse3_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 4, 1 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var upstreamGradient = new Tensor(new int[] { 4, 4 }, new double[]
            {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            });

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.TileReverse(upstreamGradient, new int[] { 1, 4 });

            // Assert
            var expectedGradient = new double[] { 10.0, 26.0, 10.0, 26.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void ReshapeReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var upstreamGradient = new Tensor(new int[] { 4 }, new double[] { 1.0, 1.0, 1.0, 1.0 });

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.ReshapeReverse(upstreamGradient, new int[] { 2, 2 });

            // Assert
            var expectedGradient = new double[] { 1.0, 1.0, 1.0, 1.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void SliceReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 3, 3 }, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var begin = new int[] { 0, 1 };
            var size = new int[] { 2, 2 };

            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.SliceReverse(upstreamGradient, begin, size);

            // Assert
            var expectedGradient = new double[] { 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void ElementwiseSquareReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.ElementwiseSquareReverse(upstreamGradient);

            // Assert
            var expectedGradient = new double[] { 2.0, 4.0, 6.0, 8.0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void ElementwiseSquareRootReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 4.0, 9.0, 16.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.ElementwiseSquareRootReverse(upstreamGradient);

            // Assert
            var expectedGradient = new double[] { 0.5, 0.25, 1.0 / 6.0, 0.125 };
            Assert.Equal(expectedGradient, gradient.Data, new DoubleArrayEqualityComparer(6d));
        }

        [Fact]
        public void ElementwiseSinReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 0.0, Math.PI / 2, Math.PI, 3 * Math.PI / 2 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.ElementwiseSinReverse(upstreamGradient);

            // Assert
            var expectedGradient = new double[] { 1.0, 0.0, -1.0, 0.0 };
            Assert.Equal(expectedGradient, gradient.Data, new DoubleArrayEqualityComparer(6d)); // 6 decimal places precision
        }

        [Fact]
        public void ElementwiseCosReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 2 }, new double[] { 0.0, Math.PI / 2, Math.PI, 3 * Math.PI / 2 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.ElementwiseCosReverse(upstreamGradient);

            // Assert
            var expectedGradient = new double[] { 0.0, -1.0, 0.0, 1.0 };
            Assert.Equal(expectedGradient, gradient.Data, new DoubleArrayEqualityComparer(6d)); // 6 decimal places precision
        }

        [Fact]
        public void ElementwiseAtan2Reverse_Test()
        {
            // Arrange
            var tensorY = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, -1.0, -1.0 });
            var tensorX = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, -1.0, -1.0, 1.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensorY });

            // Act
            var gradients = tensorReverse.ElementwiseAtan2Reverse(upstreamGradient, tensorX);

            // Assert
            var expectedGradientY = new double[] { 0.5, -0.5, -0.5, 0.5 };
            var expectedGradientX = new double[] { -0.5, -0.5, 0.5, 0.5 };
            Assert.Equal(expectedGradientY, gradients[0].Data, new DoubleArrayEqualityComparer(6d));
            Assert.Equal(expectedGradientX, gradients[1].Data, new DoubleArrayEqualityComparer(6d));
        }

        [Fact]
        public void CreateFlatArrayReverse_Test()
        {
            // Arrange
            var tensor1 = new Tensor(new int[] { 2, 3 }, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
            var tensor2 = new Tensor(new int[] { 2, 3 }, new double[] { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
            var indices = new int[] { 0, 2 };
            var upstreamGradient = new Tensor(new int[] { 8 }, new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor1, tensor2 });

            // Act
            var gradients = tensorReverse.CreateFlatArrayReverse(upstreamGradient, indices);

            // Assert
            var expectedGradient1 = new double[] { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
            var expectedGradient2 = new double[] { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
            Assert.Equal(expectedGradient1, gradients[0].Data);
            Assert.Equal(expectedGradient2, gradients[1].Data);
        }

        [Fact]
        public void UnstackReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 3, 2, 2 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            var upstreamGradients = new Tensor[]
            {
        new Tensor(new int[] { 2, 2 }, new double[] { 1,1,1,1 }),
        new Tensor(new int[] { 2, 2 }, new double[] { 2,2,2,2 }),
        new Tensor(new int[] { 2, 2 }, new double[] { 3,3,3,3 })
            };
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.UnstackReverse(upstreamGradients);

            // Assert
            var expectedGradient = new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void SumRowsReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var upstreamGradient = new Tensor(new int[] { 3, 1 }, new double[] { 1, 2, 3 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.SumRowsReverse(upstreamGradient);

            // Assert
            var expectedGradient = new double[] { 1, 1, 2, 2, 3, 3 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void TransposeReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var upstreamGradient = new Tensor(new int[] { 3, 2 }, new double[] { 1, 4, 2, 5, 3, 6 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.TransposeReverse(upstreamGradient, new int[] { 1, 0 });

            // Assert
            var expectedGradient = new double[] { 1, 2, 3, 4, 5, 6 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void ElementwiseSubReverse_Test()
        {
            // Arrange
            var tensorA = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var tensorB = new Tensor(new int[] { 2, 2 }, new double[] { 5.0, 6.0, 7.0, 8.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensorA, tensorB });

            // Act
            var gradients = tensorReverse.ElementwiseSubReverse(upstreamGradient);

            // Assert
            Assert.Equal(upstreamGradient.Data, gradients[0].Data);
            Assert.Equal(new double[] { -1.0, -1.0, -1.0, -1.0 }, gradients[1].Data);
        }

        [Fact]
        public void ElementwiseDivideReverse_Test()
        {
            // Arrange
            var tensorA = new Tensor(new int[] { 2, 2 }, new double[] { 2.0, 4.0, 6.0, 8.0 });
            var tensorB = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 2.0, 3.0, 4.0 });
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1.0, 1.0, 1.0, 1.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensorA });

            // Act
            var gradients = tensorReverse.ElementwiseDivideReverse(upstreamGradient, tensorB);

            // Assert
            Assert.Equal(new double[] { 1.0, 0.5, 1.0 / 3.0, 0.25 }, gradients[0].Data, new DoubleArrayEqualityComparer(6d));
            Assert.Equal(new double[] { -2.0, -1.0, -2.0 / 3.0, -0.5 }, gradients[1].Data, new DoubleArrayEqualityComparer(6d));
        }

        [Fact]
        public void GatherNdReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 3, 3 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var indices = new Tensor(new int[] { 2, 2 }, new double[] { 0, 0, 1, 1 });
            var upstreamGradient = new Tensor(new int[] { 2 }, new double[] { 1.0, 2.0 });
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.GatherNdReverse(upstreamGradient, indices);

            // Assert
            var expectedGradient = new double[] { 1, 0, 0, 0, 2, 0, 0, 0, 0 };
            Assert.Equal(expectedGradient, gradient.Data);
        }

        [Fact]
        public void SplitReverse_Test()
        {
            // Arrange
            var tensor = new Tensor(new int[] { 4, 3 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            var upstreamGradients = new Tensor[]
            {
                new Tensor(new int[] { 2, 3 }, new double[] { 1,1,1, 2,2,2 }),
                new Tensor(new int[] { 2, 3 }, new double[] { 3,3,3, 4,4,4 })
            };
            var tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var gradient = tensorReverse.SplitReverse(upstreamGradients);

            // Assert
            var expectedGradient = new double[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 };
            Assert.Equal(expectedGradient, gradient.Data);
        }
    }
}
