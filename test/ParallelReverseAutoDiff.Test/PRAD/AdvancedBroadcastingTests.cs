namespace ParallelReverseAutoDiff.Test.PRAD
{
    using ParallelReverseAutoDiff.PRAD;
    using System;
    using System.Linq;
    using Xunit;

    public class AdvancedBroadcastingTests
    {
        /// <summary>
        /// Helper function to compare arrays with a small tolerance for floating point differences
        /// </summary>
        private void AssertArraysEqual(double[] expected, double[] actual, double tolerance = 1e-6)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance,
                    $"Arrays differ at index {i}. Expected: {expected[i]}, Actual: {actual[i]}");
            }
        }

        #region Forward Broadcasting Tests

        [Fact]
        public void BroadcastTo_DivisibleDimensions_2To6_ReturnsCorrectResult()
        {
            // Arrange
            var tensor = new Tensor(
                new[] { 2 },
                new[] { 1.0, 2.0 }
            );
            var targetShape = new[] { 6 }; // Divisible: 6 = 2 * 3

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            AssertArraysEqual(new[] { 1.0, 2.0, 1.0, 2.0, 1.0, 2.0 }, result.Data);
        }

        [Fact]
        public void BroadcastTo_DivisibleDimensions_3To9_ReturnsCorrectResult()
        {
            // Arrange
            var tensor = new Tensor(
                new[] { 3 },
                new[] { 1.0, 2.0, 3.0 }
            );
            var targetShape = new[] { 9 }; // Divisible: 9 = 3 * 3

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            AssertArraysEqual(
                new[] { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 },
                result.Data
            );
        }

        [Fact]
        public void BroadcastTo_DivisibleDimensions_2By3To6By3_ReturnsCorrectResult()
        {
            // Arrange
            var tensor = new Tensor(
                new[] { 2, 3 },
                new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }
            );
            var targetShape = new[] { 6, 3 }; // First dimension is divisible: 6 = 2 * 3

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            AssertArraysEqual(
                new[] {
                1.0, 2.0, 3.0, // First row (original)
                4.0, 5.0, 6.0, // Second row (original)
                1.0, 2.0, 3.0, // Repeated first row
                4.0, 5.0, 6.0, // Repeated second row
                1.0, 2.0, 3.0, // Repeated first row again
                4.0, 5.0, 6.0  // Repeated second row again
                },
                result.Data
            );
        }

        [Fact]
        public void BroadcastTo_DivisibleDimensions_2By2To2By6_ReturnsCorrectResult()
        {
            // Arrange
            var tensor = new Tensor(
                new[] { 2, 2 },
                new[] { 1.0, 2.0, 3.0, 4.0 }
            );
            var targetShape = new[] { 2, 6 }; // Second dimension is divisible: 6 = 2 * 3

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            AssertArraysEqual(
                new[] {
                1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // First row repeated 3 times
                3.0, 4.0, 3.0, 4.0, 3.0, 4.0  // Second row repeated 3 times
                },
                result.Data
            );
        }

        [Fact]
        public void BroadcastTo_MultipleDivisibleDimensions_ReturnsCorrectResult()
        {
            // Arrange
            var tensor = new Tensor(
                new[] { 2, 3 },
                new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }
            );
            var targetShape = new[] { 6, 9 }; // Both dimensions divisible: 6 = 2 * 3, 9 = 3 * 3

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            Assert.Equal(6 * 9, result.Data.Length);

            // Check the first row repeated pattern
            for (int i = 0; i < 6; i += 2)
            {
                for (int j = 0; j < 9; j += 3)
                {
                    // First row of original tensor
                    Assert.Equal(1.0, result.Data[i * 9 + j]);
                    Assert.Equal(2.0, result.Data[i * 9 + j + 1]);
                    Assert.Equal(3.0, result.Data[i * 9 + j + 2]);

                    // Second row of original tensor
                    Assert.Equal(4.0, result.Data[(i + 1) * 9 + j]);
                    Assert.Equal(5.0, result.Data[(i + 1) * 9 + j + 1]);
                    Assert.Equal(6.0, result.Data[(i + 1) * 9 + j + 2]);
                }
            }
        }

        [Fact]
        public void BroadcastTo_MixedBroadcastingTypes_ReturnsCorrectResult()
        {
            // Arrange
            var tensor = new Tensor(
                new[] { 1, 1, 6 },
                new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }
            );
            var targetShape = new[] { 6, 4, 12 };
            // First dim: 6 = 1 * 6 (divisible)
            // Second dim: 4 = 1 ^ 4 (divisible)
            // Third dim: 12 = 6 * 2 (divisible)

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            Assert.Equal(6 * 4 * 12, result.Data.Length);

            int idx = 5 * 48 + 3;
            Assert.Equal(4.0, result.Data[idx]);
            Assert.Equal(5.0, result.Data[idx + 1]);
            Assert.Equal(6.0, result.Data[idx + 2]);
        }

        [Fact]
        public void BroadcastTo_LargeDivisibleCase_ReturnsCorrectResult()
        {
            // Arrange - similar to the mentioned 35 to 4095 case
            int originalSize = 35;
            int targetSize = 35 * 117; // 4095

            // Create sequential data for easy verification
            double[] data = new double[originalSize];
            for (int i = 0; i < originalSize; i++)
            {
                data[i] = i + 1;
            }

            var tensor = new Tensor(new[] { originalSize }, data);
            var targetShape = new[] { targetSize };

            // Act
            var result = tensor.BroadcastTo(targetShape);

            // Assert
            Assert.Equal(targetShape, result.Shape);
            Assert.Equal(targetSize, result.Data.Length);

            // Check that the pattern repeats correctly
            for (int i = 0; i < 117; i++)
            {
                for (int j = 0; j < originalSize; j++)
                {
                    Assert.Equal(j + 1, result.Data[i * originalSize + j]);
                }
            }
        }

        #endregion

        #region Reverse Broadcasting Tests

        [Fact]
        public void BroadcastToReverse_StandardCase_SumsGradients()
        {
            // Arrange
            var originalShape = new[] { 2 };
            var broadcastShape = new[] { 6 }; // 6 = 2 * 3

            // Create a tensor with the original shape
            var tensor = new Tensor(originalShape, new double[] { 1.0, 2.0 });

            // Create a gradient tensor in the broadcasted shape
            var gradient = new Tensor(broadcastShape, new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var result = tensorReverse.BroadcastToReverse(gradient, originalShape);

            // Assert
            Assert.Equal(originalShape, result.Shape);
            // Each element in the original should get the sum of its repeated values
            // First element: 0.1 + 0.3 + 0.5 = 0.9
            // Second element: 0.2 + 0.4 + 0.6 = 1.2
            AssertArraysEqual(new double[] { 0.9, 1.2 }, result.Data);
        }

        [Fact]
        public void BroadcastToReverse_MatrixCase_SumsGradients()
        {
            // Arrange
            var originalShape = new[] { 2, 2 };
            var broadcastShape = new[] { 2, 6 }; // 6 = 2 * 3

            // Create a tensor with the original shape
            var tensor = new Tensor(originalShape, new double[] { 1.0, 2.0, 3.0, 4.0 });

            // Create a gradient tensor in the broadcasted shape (2x6)
            var gradientData = new double[] {
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  // First row (original values repeated 3 times)
            0.7, 0.8, 0.9, 1.0, 1.1, 1.2   // Second row (original values repeated 3 times)
        };
            var gradient = new Tensor(broadcastShape, gradientData);

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var result = tensorReverse.BroadcastToReverse(gradient, originalShape);

            // Assert
            Assert.Equal(originalShape, result.Shape);

            // Expected values:
            // First element (0,0): 0.1 + 0.3 + 0.5 = 0.9
            // Second element (0,1): 0.2 + 0.4 + 0.6 = 1.2
            // Third element (1,0): 0.7 + 0.9 + 1.1 = 2.7
            // Fourth element (1,1): 0.8 + 1.0 + 1.2 = 3.0
            AssertArraysEqual(new double[] { 0.9, 1.2, 2.7, 3.0 }, result.Data);
        }

        [Fact]
        public void BroadcastToReverse_MultipleAxisBroadcast_SumsGradients()
        {
            // Arrange
            var originalShape = new[] { 2, 3 };
            var broadcastShape = new[] { 4, 2, 3 }; // Broadcasting by adding a dimension at the front

            // Create a tensor with the original shape
            var tensor = new Tensor(originalShape, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Create a gradient tensor in the broadcasted shape (4x2x3)
            // We'll use values that make it easy to verify the sums
            double[] gradientData = new double[24]; // 4 * 2 * 3
            for (int i = 0; i < 24; i++)
            {
                gradientData[i] = 0.1 * (i + 1);
            }
            var gradient = new Tensor(broadcastShape, gradientData);

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var result = tensorReverse.BroadcastToReverse(gradient, originalShape);

            // Assert
            Assert.Equal(originalShape, result.Shape);

            // Expected results - each element gets sum of 4 values
            double[] expected = new double[6];
            for (int i = 0; i < 6; i++)
            {
                expected[i] = 0;
                for (int j = 0; j < 4; j++)
                {
                    expected[i] += 0.1 * (j * 6 + i + 1);
                }
            }

            AssertArraysEqual(expected, result.Data);
        }

        [Fact]
        public void BroadcastToReverse_MixedBroadcasting_SumsGradients()
        {
            // Arrange - similar to forward test
            var originalShape = new[] { 2, 1, 3 };
            var broadcastShape = new[] { 6, 4, 9 };
            // First dim: 6 = 2 * 3 (divisible) - adds factor of 3
            // Second dim: 4 (from 1) - adds factor of 4
            // Third dim: 9 = 3 * 3 (divisible) - adds factor of 3
            // Total factor: 3 * 4 * 3 = 36

            // Create a tensor with the original shape
            var tensor = new Tensor(originalShape, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Create a gradient tensor with constant values for easy verification
            double[] gradientData = new double[6 * 4 * 9];
            for (int i = 0; i < gradientData.Length; i++)
            {
                gradientData[i] = 0.1;
            }
            var gradient = new Tensor(broadcastShape, gradientData);

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var result = tensorReverse.BroadcastToReverse(gradient, originalShape);

            // Assert
            Assert.Equal(originalShape, result.Shape);

            // Each original value is repeated 36 times (3*4*3), each with value 0.1
            // So each element should sum to 36 * 0.1 = 3.6
            AssertArraysEqual(new double[] { 3.6, 3.6, 3.6, 3.6, 3.6, 3.6 }, result.Data);
        }

        [Fact]
        public void BroadcastToReverse_LargeDivisibleCase_SumsGradients()
        {
            // Arrange - similar to the mentioned 35 to 4095 case
            int originalSize = 35;
            int targetSize = 35 * 117; // 4095

            // Create a tensor with original shape
            double[] data = new double[originalSize];
            for (int i = 0; i < originalSize; i++)
            {
                data[i] = i + 1;
            }
            var tensor = new Tensor(new[] { originalSize }, data);

            // Create gradient with constant values for easy verification
            double[] gradientData = new double[targetSize];
            for (int i = 0; i < gradientData.Length; i++)
            {
                gradientData[i] = 0.01;
            }
            var gradient = new Tensor(new[] { targetSize }, gradientData);

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var result = tensorReverse.BroadcastToReverse(gradient, new[] { originalSize });

            // Assert
            Assert.Equal(new[] { originalSize }, result.Shape);

            // Each original element is repeated 117 times, each with value 0.01
            // So each should sum to 117 * 0.01 = 1.17
            double[] expected = new double[originalSize];
            for (int i = 0; i < originalSize; i++)
            {
                expected[i] = 1.17;
            }

            AssertArraysEqual(expected, result.Data);
        }

        [Fact]
        public void BroadcastToReverse_GradientMatchingOriginal_Passthrough()
        {
            // Arrange - No actual broadcasting, shapes match
            var originalShape = new[] { 2, 3 };

            // Create a tensor with the original shape
            var tensor = new Tensor(originalShape, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Create a gradient tensor with the same shape
            var gradient = new Tensor(originalShape, new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act
            var result = tensorReverse.BroadcastToReverse(gradient, originalShape);

            // Assert
            Assert.Equal(originalShape, result.Shape);
            // When shapes match, gradients should pass through unchanged
            AssertArraysEqual(gradient.Data, result.Data);
        }

        #endregion

        #region Roundtrip Tests

        [Fact]
        public void BroadcastAndReverse_RoundTrip_CorrectGradients()
        {
            // Arrange
            var originalShape = new[] { 2, 3 };
            var broadcastShape = new[] { 4, 2, 3 };

            // Create original tensor
            var tensor = new Tensor(originalShape, new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Forward pass - broadcast the tensor
            var broadcasted = tensor.BroadcastTo(broadcastShape);

            // Create a gradient in the broadcasted shape
            // We'll use the output values as gradients (just for this test)
            var upstreamGradient = new Tensor(broadcastShape, broadcasted.Data);

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act - backward pass
            var inputGradient = tensorReverse.BroadcastToReverse(upstreamGradient, originalShape);

            // Assert
            Assert.Equal(originalShape, inputGradient.Shape);

            // In this case, each element was broadcast to 4 identical copies
            // So each gradient should be 4 times the original value
            double[] expected = new double[6];
            for (int i = 0; i < 6; i++)
            {
                expected[i] = tensor.Data[i] * 4;
            }

            AssertArraysEqual(expected, inputGradient.Data);
        }

        [Fact]
        public void BroadcastAndReverse_ComplexCase_CorrectGradients()
        {
            // This test simulates a more complex scenario with mixed broadcasting types

            // Arrange
            var originalShape = new[] { 2, 1, 2 };
            var broadcastShape = new[] { 4, 2, 3, 4 }; // Complex broadcasting

            // Create original tensor
            var originalData = new double[] { 1.0, 2.0, 3.0, 4.0 };
            var tensor = new Tensor(originalShape, originalData);

            // Forward pass - broadcast the tensor
            var broadcasted = tensor.BroadcastTo(broadcastShape);

            // Create uniform gradient
            var gradientData = new double[4 * 2 * 3 * 4];
            for (int i = 0; i < gradientData.Length; i++)
            {
                gradientData[i] = 1.0;
            }
            var gradient = new Tensor(broadcastShape, gradientData);

            TensorReverse tensorReverse = new TensorReverse(new Tensor[] { tensor });

            // Act - backward pass
            var result = tensorReverse.BroadcastToReverse(gradient, originalShape);

            // Assert
            Assert.Equal(originalShape, result.Shape);

            // Calculate expected values:
            // Each value from the original tensor gets repeated:
            // - 4 times across first dimension
            // - 3 times across third dimension
            // - 2 times across fourth dimension (2 to 4)
            // Total: 4 * 3 * 2 = 24 times
            double[] expected = new double[4];
            for (int i = 0; i < 4; i++)
            {
                expected[i] = 24.0; // Each gradient is 1.0, summed 24 times
            }

            AssertArraysEqual(expected, result.Data);
        }

        #endregion
    }
}
