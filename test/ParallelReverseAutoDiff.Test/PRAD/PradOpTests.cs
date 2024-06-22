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

        [Fact]
        public void TestStack()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var tensorToStack = new Tensor(new int[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Stack(new Tensor[] { tensorToStack }, axis: 0);

            Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2, 2 }, new double[] { 1, 1, 1, 1, 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[0].Data);
            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[1].Data);
        }

        [Fact]
        public void TestConcat()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var tensorToConcat = new Tensor(new int[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
            var pradOp = new PradOp(seed);

            var result = pradOp.Concat(new Tensor[] { tensorToConcat }, axis: 0);

            Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 4, 2 }, new double[] { 1, 1, 1, 1, 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[0].Data);
            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[1].Data);
        }

        [Fact]
        public void TestComplexComposition()
        {
            var seed = new Tensor(new int[] { 3, 3, 3 }, new double[] {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27
    });
            var pradOp = new PradOp(seed);

            // Reshape to 9x3
            var reshaped = pradOp.Reshape(new int[] { 9, 3 });

            // Tile 2x on the first dimension
            var tiled = pradOp.Tile(new int[] { 2, 1 });
            string tiledCode = tiled.Result.PrintCode();

            // Gather specific indices
            var indices = new Tensor(new int[] { 3 }, new double[] { 0, 8, 16 });
            var gathered = pradOp.Gather(indices, axis: 0);

            // Assert the shape of the result
            Assert.Equal(new int[] { 3, 3 }, gathered.Result.Shape);

            // Assert specific values
            Assert.Equal(1, gathered.Result.Data[0]);
            Assert.Equal(2, gathered.Result.Data[1]);
            Assert.Equal(3, gathered.Result.Data[2]);

            Assert.Equal(25, gathered.Result.Data[3]);
            Assert.Equal(26, gathered.Result.Data[4]);
            Assert.Equal(27, gathered.Result.Data[5]);

            Assert.Equal(22, gathered.Result.Data[6]);
            Assert.Equal(23, gathered.Result.Data[7]);
            Assert.Equal(24, gathered.Result.Data[8]);

            // Print the entire result for debugging
            Console.WriteLine($"Result shape: [{string.Join(", ", gathered.Result.Shape)}]");
            Console.WriteLine($"Result data: [{string.Join(", ", gathered.Result.Data)}]");
        }

        [Fact]
        public void TestLargeTensorOperations()
        {
            var largeShape = new int[] { 1000, 1000 };
            var seed = new Tensor(largeShape, Enumerable.Range(0, 1000000).Select(i => (double)i).ToArray());
            var pradOp = new PradOp(seed);

            var result = pradOp.Add(seed);  // Add the tensor to itself

            Assert.Equal(1000000, result.Result.Data.Length);
            Assert.Equal(0, result.Result.Data[0]);
            Assert.Equal(1999998, result.Result.Data[999999]);

            var upstreamGradient = new Tensor(largeShape, Enumerable.Repeat(1.0, 1000000).ToArray());
            pradOp.Back(upstreamGradient);

            Assert.Equal(1, result.Gradients[0].Data[0]);
            Assert.Equal(1, result.Gradients[0].Data[999999]);
        }

        [Fact]
        public void TestComplexReshape()
        {
            var seed = new Tensor(new int[] { 2, 3, 4 }, Enumerable.Range(1, 24).Select(i => (double)i).ToArray());
            var pradOp = new PradOp(seed);

            // Reshape to 1D
            var reshaped1D = pradOp.Reshape(new int[] { 24 });
            Assert.Equal(new int[] { 24 }, reshaped1D.Result.Shape);

            // Reshape to 4D
            var reshaped4D = pradOp.Reshape(new int[] { 2, 2, 2, 3 });
            Assert.Equal(new int[] { 2, 2, 2, 3 }, reshaped4D.Result.Shape);

            // Reshape with -1
            var reshapedAuto = pradOp.Reshape(new int[] { -1, 6 });
            Assert.Equal(new int[] { 4, 6 }, reshapedAuto.Result.Shape);

            // Backpropagate through all reshapes
            var upstreamGradient = new Tensor(new int[] { 4, 6 }, Enumerable.Repeat(1.0, 24).ToArray());
            pradOp.Back(upstreamGradient);

            Assert.Equal(Enumerable.Repeat(1.0, 24).ToArray(), reshapedAuto.Gradients[0].Data);
        }

        [Fact]
        public void TestSimpleGatherAlongAxis1()
        {
            var seed = new Tensor(new int[] { 2, 3, 2 }, new double[] {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12
    });
            var pradOp = new PradOp(seed);

            var indices = new Tensor(new int[] { 2 }, new double[] { 0, 2 });
            var gathered = pradOp.Gather(indices, axis: 1);

            Assert.Equal(new int[] { 2, 2, 2 }, gathered.Result.Shape);
            Assert.Equal(new double[] { 1, 2, 5, 6, 7, 8, 11, 12 }, gathered.Result.Data);
        }

        [Fact]
        public void TestComplexGather()
        {
            var seed = new Tensor(new int[] { 3, 4, 5 }, Enumerable.Range(0, 60).Select(i => (double)i).ToArray());
            var pradOp = new PradOp(seed);

            // Gather along axis 0
            var indices0 = new Tensor(new int[] { 2 }, new double[] { 1, 2 });
            var gathered0 = pradOp.Gather(indices0, axis: 0);
            Assert.Equal(new int[] { 2, 4, 5 }, gathered0.Result.Shape);

            // Gather along axis 1
            var indices1 = new Tensor(new int[] { 3 }, new double[] { 0, 2, 3 });
            var gathered1 = pradOp.Gather(indices1, axis: 1);
            Assert.Equal(new int[] { 2, 3, 5 }, gathered1.Result.Shape);

            // Gather along axis 2
            var indices2 = new Tensor(new int[] { 2 }, new double[] { 1, 3 });
            var gathered2 = pradOp.Gather(indices2, axis: 2);
            Assert.Equal(new int[] { 2, 3, 2 }, gathered2.Result.Shape);

            // Backpropagate through one of the gathers
            var upstreamGradient = new Tensor(gathered1.Result.Shape, Enumerable.Repeat(1.0, 2 * 3 * 5).ToArray());
            pradOp.Back(upstreamGradient);

            // Check that gradients are distributed correctly
            var expectedGradientSum = 2 * 3 * 5;  // Sum of all 1s in upstream gradient
            Assert.Equal(expectedGradientSum, gathered1.Gradients[0].Data.Sum());
        }

        [Fact]
        public void TestChainedGather()
        {
            var seed = new Tensor(new int[] { 3, 4, 5 }, Enumerable.Range(0, 60).Select(i => (double)i).ToArray());
            var pradOp = new PradOp(seed);

            // First Gather along axis 0
            var indices0 = new Tensor(new int[] { 2 }, new double[] { 1, 2 });
            var gathered0 = pradOp.Gather(indices0, axis: 0);
            string shape0 = string.Join(", ", gathered0.Result.Shape);
            string result0 = string.Join(", ", gathered0.Result.Data);
            Console.WriteLine($"First Gather shape: [{shape0}]");
            Console.WriteLine($"First Gather data: [{result0}]");

            // Log the state of PradOp after first Gather
            Console.WriteLine($"PradOp state after first Gather: {pradOp}");

            // Second Gather along axis 1
            var indices1 = new Tensor(new int[] { 3 }, new double[] { 0, 2, 3 });
            string code = pradOp.PrintCodeForCurrentTensor();
            var gathered1 = pradOp.Gather(indices1, axis: 1);
            string shape1 = string.Join(", ", gathered1.Result.Shape);
            string result1 = string.Join(", ", gathered1.Result.Data);
            Console.WriteLine($"shape1: {shape1}");
            Console.WriteLine($"result1: {result1}");

            // Verify some expected values
            Console.WriteLine($"First element: {gathered1.Result.Data[0]}");
            Console.WriteLine($"Element at index 5: {gathered1.Result.Data[5]}");
            Console.WriteLine($"Element at index 15: {gathered1.Result.Data[15]}");

            Assert.Equal(20, gathered1.Result.Data[0]);
            Assert.Equal(30, gathered1.Result.Data[5]);
            Assert.Equal(40, gathered1.Result.Data[15]);
        }

        [Fact]
        public void TestGatherOnSpecificTensor()
        {
            // Create the specific tensor
            var tensor = new Tensor(new int[] { 2, 4, 5 }, new double[] {
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
    });

            // Create PradOp instance
            var pradOp = new PradOp(tensor);

            // Create indices for gathering along axis 1
            var indices = new Tensor(new int[] { 3 }, new double[] { 0, 2, 3 });

            // Perform Gather operation
            var result = pradOp.Gather(indices, axis: 1);

            // Assert the shape of the result
            Assert.Equal(new int[] { 2, 3, 5 }, result.Result.Shape);

            // Assert specific values
            // First batch
            // First slice (index 0)
            Assert.Equal(20, result.Result.Data[0]);
            Assert.Equal(21, result.Result.Data[1]);
            Assert.Equal(22, result.Result.Data[2]);
            Assert.Equal(23, result.Result.Data[3]);
            Assert.Equal(24, result.Result.Data[4]);
            // Second slice (index 2)
            Assert.Equal(30, result.Result.Data[5]);
            Assert.Equal(31, result.Result.Data[6]);
            Assert.Equal(32, result.Result.Data[7]);
            Assert.Equal(33, result.Result.Data[8]);
            Assert.Equal(34, result.Result.Data[9]);
            // Third slice (index 3)
            Assert.Equal(35, result.Result.Data[10]);
            Assert.Equal(36, result.Result.Data[11]);
            Assert.Equal(37, result.Result.Data[12]);
            Assert.Equal(38, result.Result.Data[13]);
            Assert.Equal(39, result.Result.Data[14]);

            // Second batch
            // First slice (index 0)
            Assert.Equal(40, result.Result.Data[15]);
            Assert.Equal(41, result.Result.Data[16]);
            Assert.Equal(42, result.Result.Data[17]);
            Assert.Equal(43, result.Result.Data[18]);
            Assert.Equal(44, result.Result.Data[19]);
            // Second slice (index 2)
            Assert.Equal(50, result.Result.Data[20]);
            Assert.Equal(51, result.Result.Data[21]);
            Assert.Equal(52, result.Result.Data[22]);
            Assert.Equal(53, result.Result.Data[23]);
            Assert.Equal(54, result.Result.Data[24]);
            // Third slice (index 3)
            Assert.Equal(55, result.Result.Data[25]);
            Assert.Equal(56, result.Result.Data[26]);
            Assert.Equal(57, result.Result.Data[27]);
            Assert.Equal(58, result.Result.Data[28]);
            Assert.Equal(59, result.Result.Data[29]);

            // Print the entire result for debugging
            Console.WriteLine($"Result shape: [{string.Join(", ", result.Result.Shape)}]");
            Console.WriteLine($"Result data: [{string.Join(", ", result.Result.Data)}]");
        }

        [Fact]
        public void TestComplexGather2()
        {
            var seed = new Tensor(new int[] { 3, 4, 5 }, Enumerable.Range(0, 60).Select(i => (double)i).ToArray());
            var pradOp = new PradOp(seed);

            // Gather along axis 1
            var indices1 = new Tensor(new int[] { 3 }, new double[] { 0, 2, 3 });
            var gathered1 = pradOp.Gather(indices1, axis: 1);

            Console.WriteLine($"Gathered shape: [{string.Join(", ", gathered1.Result.Shape)}]");
            Console.WriteLine($"Gathered data (first 20 elements): [{string.Join(", ", gathered1.Result.Data.Take(20))}]");

            Assert.Equal(new int[] { 3, 3, 5 }, gathered1.Result.Shape);

            // Add more specific assertions based on expected values
            // For example, check the first few elements of the gathered tensor
            Assert.Equal(0, gathered1.Result.Data[0]);
            Assert.Equal(10, gathered1.Result.Data[5]);
            Assert.Equal(15, gathered1.Result.Data[10]);
        }

        [Fact]
        public void TestIntermediateGather()
        {
            var seed = new Tensor(new int[] { 2, 3, 3 }, new double[] {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18
    });
            var pradOp = new PradOp(seed);

            var indices = new Tensor(new int[] { 2 }, new double[] { 0, 2 });
            var gathered = pradOp.Gather(indices, axis: 1);

            Console.WriteLine($"Intermediate gathered shape: [{string.Join(", ", gathered.Result.Shape)}]");
            Console.WriteLine($"Intermediate gathered data: [{string.Join(", ", gathered.Result.Data)}]");

            Assert.Equal(new int[] { 2, 2, 3 }, gathered.Result.Shape);
            Assert.Equal(new double[] { 1, 2, 3, 7, 8, 9, 10, 11, 12, 16, 17, 18 }, gathered.Result.Data);
        }

        [Fact]
        public void TestComplexTile()
        {
            var seed = new Tensor(new int[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var pradOp = new PradOp(seed);

            var tiled = pradOp.Tile(new int[] { 3, 2 });
            Assert.Equal(new int[] { 6, 6 }, tiled.Result.Shape);
            Assert.Equal(36, tiled.Result.Data.Length);

            // Check first row of tiled tensor
            Assert.Equal(new double[] { 1, 2, 3, 1, 2, 3 }, tiled.Result.Data.Take(6).ToArray());

            // Backpropagate
            var upstreamGradient = new Tensor(new int[] { 6, 6 }, Enumerable.Repeat(1.0, 36).ToArray());
            pradOp.Back(upstreamGradient);

            // Each original element should receive a gradient equal to the number of times it was tiled
            Assert.Equal(new double[] { 6, 6, 6, 6, 6, 6 }, tiled.Gradients[0].Data);
        }

        [Fact]
        public void TestStackAndConcatMany()
        {
            var tensors = Enumerable.Range(0, 10)
                .Select(i => new Tensor(new int[] { 10, 2, 2 },
                    Enumerable.Range(0, 40).Select(j => (double)(i * 40 + j + 1)).ToArray()))
                .ToArray();
            var pradOp = new PradOp(tensors[0]);

            // Test Stack
            var stacked = pradOp.Stack(tensors.Skip(1).ToArray(), axis: 0);
            Assert.Equal(new int[] { 10, 10, 2, 2 }, stacked.Result.Shape);

            // We can't directly concat the stacked result with the original tensors
            // Instead, let's reshape the stacked result to make it compatible
            var reshaped = pradOp.Reshape(new int[] { 100, 2, 2 });
            Assert.Equal(new int[] { 100, 2, 2 }, reshaped.Result.Shape);

            // Now we can concat the reshaped result with the original first tensor
            var concatenated = pradOp.Concat(new[] { tensors[0] }, axis: 0);
            Assert.Equal(new int[] { 110, 2, 2 }, concatenated.Result.Shape);

            // Backpropagate through all operations
            var upstreamGradient = new Tensor(new int[] { 110, 2, 2 }, Enumerable.Repeat(1.0, 440).ToArray());
            pradOp.Back(upstreamGradient);

            // Check gradients
            // The first tensor participates in concat only
            Assert.Equal(400, concatenated.Gradients[0].Data.Length);
            Assert.Equal(40, concatenated.Gradients[1].Data.Length);
            Assert.All(concatenated.Gradients[0].Data, grad => Assert.Equal(1.0, grad));

            // The other tensors participate in stack, reshape, and concat
            for (int i = 1; i < concatenated.Gradients.Length; i++)
            {
                Assert.Equal(40, concatenated.Gradients[i].Data.Length);
                Assert.All(concatenated.Gradients[i].Data, grad => Assert.Equal(1.0, grad));
            }
        }

        [Fact]
        public void TestCustomOperation()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            var pradOp = new PradOp(seed);

            // Define a custom operation (e.g., element-wise square)
            Func<Tensor, Tensor> operation = tensor => tensor.ElementwiseSquare();
            Func<Tensor, Tensor, Tensor, Tensor[]> reverseOperation = (input, output, upstreamGrad) =>
            {
                var grad = new Tensor(input.Shape);
                for (int i = 0; i < input.Data.Length; i++)
                {
                    grad.Data[i] = 2 * input.Data[i] * upstreamGrad.Data[i];
                }
                return new Tensor[] { grad };
            };

            var result = pradOp.CustomOperation(operation, reverseOperation, 1, new int[] { 2, 2 });

            Assert.Equal(new double[] { 1, 4, 9, 16 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 2, 4, 6, 8 }, result.Gradients[0].Data);
        }
    }
}
