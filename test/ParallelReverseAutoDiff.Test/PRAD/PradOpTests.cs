using ILGPU.Runtime.Cuda;
using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using System.Diagnostics;
using Xunit;
using static ParallelReverseAutoDiff.PRAD.PradOp;

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
        public void TestMassiveVNNOperation()
        {
            // Create massive tensors (2000x4000)
            var seed = new Tensor(new int[] { 2000, 4000 }, Enumerable.Range(0, 8000000).Select(i => (double)i).ToArray());
            var other = new Tensor(new int[] { 2000, 4000 }, Enumerable.Range(0, 8000000).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 2000, 2000 }, Enumerable.Range(0, 4000000).Select(i => (double)(i % 10)).ToArray());

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            var opSeed = new PradOp(seed);
            var opOther = new PradOp(other);

            var clonedOpSeed = opSeed.DeepClone();
            var clonedOther = opOther.DeepClone();

            // Slice operations
            var anglesSeed = opSeed.Slice(new int[] { 0, 2000 }, new int[] { 2000, 2000 }).Result;
            var anglesOther = opOther.Slice(new int[] { 0, 2000 }, new int[] { 2000, 2000 }).Result;

            // Concatenation
            var concatAngles = opSeed.Concat(new[] { anglesOther }, axis: 1).Result;

            // Reshape
            var flatAngles = opSeed.Reshape(new int[] { 1, 8000000 }).Result;

            // Trigonometric operations
            var clonedFlatAnglesOp = opSeed.DeepClone();
            var sinAngles = opSeed.Sin().Result;
            var cosAngles = clonedFlatAnglesOp.Cos().Result;

            // More slicing
            var magnitudesSeed = clonedOpSeed.Slice(new int[] { 0, 0 }, new int[] { 2000, 2000 }).Result;
            var magnitudesOther = clonedOther.Slice(new int[] { 0, 0 }, new int[] { 2000, 2000 }).Result;

            // Another concatenation
            var concatMagnitudes = clonedOpSeed.Concat(new[] { magnitudesOther }, axis: 1).Result;

            // Another reshape
            var flatMagnitudes = clonedOpSeed.Reshape(new int[] { 1, 8000000 }).Result;

            // Multiplication operations
            var clonedFlatMagnitudesOp = clonedOpSeed.DeepClone();
            var Ys = clonedOpSeed.Mul(sinAngles).Result;
            var Xs = clonedFlatMagnitudesOp.Mul(cosAngles).Result;

            stopwatch.Stop();
            Console.WriteLine($"Total execution time: {stopwatch.ElapsedMilliseconds} ms");

            // Assertions
            Assert.Equal(new int[] { 2000, 4000 }, concatAngles.Shape);
            Assert.Equal(new int[] { 1, 8000000 }, flatAngles.Shape);
            Assert.Equal(new int[] { 1, 8000000 }, sinAngles.Shape);
            Assert.Equal(new int[] { 1, 8000000 }, cosAngles.Shape);
            Assert.Equal(new int[] { 2000, 4000 }, concatMagnitudes.Shape);
            Assert.Equal(new int[] { 1, 8000000 }, flatMagnitudes.Shape);
            Assert.Equal(new int[] { 1, 8000000 }, Ys.Shape);
            Assert.Equal(new int[] { 1, 8000000 }, Xs.Shape);
        }

        [Fact]
        public void TestMultipleBranches()
        {
            Matrix input1 = new Matrix(100, 200);
            input1.Initialize(InitializationType.Xavier);

            Matrix input2 = new Matrix(100, 200);
            input2.Initialize(InitializationType.Xavier);

            var pradOp1 = new PradOp(input1.ToTensor());
            var pradOp2 = new PradOp(input2.ToTensor());

            var sinCosRes1 = pradOp1.Sin()
                .Then(PradOp.CosOp);

            var sinCosRes2 = pradOp2.Sin()
                .Then(PradOp.CosOp);

            var addRes = sinCosRes1.PradOp.Add(sinCosRes2.Result);

            Matrix gradientOfTheLoss = new Matrix(100, 200);
            gradientOfTheLoss.Initialize(InitializationType.Xavier);

            addRes.Back(gradientOfTheLoss.ToTensor());

            Assert.NotNull(pradOp1.SeedGradient);
            Assert.NotNull(pradOp2.SeedGradient);
            Assert.Equal(pradOp1.SeedGradient.Shape, input1.Shape);
            Assert.Equal(pradOp2.SeedGradient.Shape, input2.Shape);
        }

        [Fact]
        public void TestDoParallel()
        {
            Matrix input1 = new Matrix(100, 200);
            input1.Initialize(InitializationType.Xavier);

            var pradOp = new PradOp(input1.ToTensor());

            var (sinOp, cosOp) = pradOp.DoParallel(
                op => op.Sin(),
                op => op.Cos());

            var addOp = sinOp.PradOp.Add(cosOp.Result);

            Matrix gradientOfTheLoss = new Matrix(100, 200);
            gradientOfTheLoss.Initialize(InitializationType.Xavier);

            pradOp.Back(gradientOfTheLoss.ToTensor());

            Assert.NotNull(pradOp.SeedGradient);
            Assert.Equal(pradOp.SeedGradient.Shape, input1.Shape);
        }

        [Fact]
        public void TestBranch()
        {
            Matrix input1 = new Matrix(200, 200);
            input1.Initialize(InitializationType.Xavier);

            var pradOp = new PradOp(input1.ToTensor());

            var (anglesCos, anglesSin) = pradOp.DoParallel(
                a => a.Cos(),
                a => a.Sin());

            anglesCos.Then(PradOp.AddOp, anglesSin.Result)
                .Then(PradOp.SquareOp);

            var anotherBranch = anglesCos.Branch();

            var anotherResult = anotherBranch.Transpose(new int[] { 1, 0 });

            anglesCos.Then(PradOp.AddOp, anotherResult.Result);

            var gradientOfTheLoss = new Matrix(200, 200);
            gradientOfTheLoss.Initialize(InitializationType.Xavier);

            pradOp.Back(gradientOfTheLoss.ToTensor());

            Assert.NotNull(pradOp.SeedGradient);
            Assert.Equal(pradOp.SeedGradient.Shape, input1.Shape);
        }

        [Fact]
        public void TestDoParallel2()
        {
            Matrix input1 = new Matrix(100, 200);
            input1.Initialize(InitializationType.Xavier);

            var pradOp = new PradOp(input1.ToTensor());

            var (anglesCos, anglesSin) = pradOp.DoParallel(
                a => a.Cos(),
                a => a.Sin());

            anglesCos.Then(PradOp.AddOp, anglesSin.Result)
                .Then(PradOp.SquareOp);

            var gradientOfTheLoss = new Matrix(100, 200);
            gradientOfTheLoss.Initialize(InitializationType.Xavier);

            pradOp.Back(gradientOfTheLoss.ToTensor());

            Assert.NotNull(pradOp.SeedGradient);
            Assert.Equal(pradOp.SeedGradient.Shape, input1.Shape);
        }

        [Fact]
        public void TestSplitAndBack()
        {
            Matrix input1 = new Matrix(100, 200);
            input1.Initialize(InitializationType.Xavier);

            var pradOp = new PradOp(input1.ToTensor());

            var half = input1.Cols / 2;

            var (magnitudes, angles) = pradOp.Split(half, 1);

            var (anglesCos, anglesSin) = angles.DoParallel(
                a => a.Cos(),
                a => a.Sin());

            var (x, y) = magnitudes.DoParallel(
                m => m.Mul(anglesCos.Result),
                m => m.Mul(anglesSin.Result));

            x.Then(PradOp.AddOp, y.Result)
             .Then(PradOp.SquareOp);
            var gradientOfTheLoss = new Matrix(100, 100);
            gradientOfTheLoss.Initialize(InitializationType.Xavier);

            x.Back(gradientOfTheLoss.ToTensor());

            Assert.NotNull(pradOp.SeedGradient);
            Assert.Equal(pradOp.SeedGradient.Shape, input1.Shape);
        }

        [Fact]
        public void TestLessThanWhereAndModulus()
        {
            // Create input tensors
            var x = new Tensor(new int[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var y = new Tensor(new int[] { 2, 3 }, new double[] { 3, 3, 3, 3, 3, 3 });
            var pradOp = new PradOp(x);

            // Perform operations
            var result = pradOp
                .LessThan(y)  // Check which elements of x are less than 3
                .Then(lessThanResult => {
                    var lessThanResultBranch = lessThanResult.PradOp.Branch();
                    var modulusResult = lessThanResult.PradOp.Modulus(new Tensor(new int[] { 2, 3 }, new double[] { 2, 2, 2, 2, 2, 2 }));
                    return modulusResult.PradOp.Where(lessThanResultBranch.BranchInitialTensor, y);
                });

            // Compute gradients
            var upstreamGradient = new Tensor(new int[] { 2, 3 }, new double[] { 1, 1, 1, 1, 1, 1 });
            GradientRecorder.Instance.RecordingEnabled = true;
            var gradient = pradOp.Back(upstreamGradient);
        }

        [Fact]
        public void TestElementwiseVectorAddOperation()
        {
            Matrix input1 = new Matrix(100, 200);
            input1.Initialize(InitializationType.Xavier);

            Matrix input2 = new Matrix(100, 200);
            input2.Initialize(InitializationType.He);

            var pradOp = new PradOp(input1.ToTensor());
            var pradOp2 = new PradOp(input2.ToTensor());

            var half = input1.Cols / 2;

            var (magnitudes1, angles1) = pradOp.Split(half, axis: 1);
            var (magnitudes2, angles2) = pradOp2.Split(half, axis: 1);

            var (x1, y1) = magnitudes1.DoParallel(
                m => m.Mul(angles1.Cos().Result),
                m => m.Mul(angles1.Sin().Result));

            var (x2, y2) = magnitudes2.DoParallel(
                m => m.Mul(angles2.Cos().Result),
                m => m.Mul(angles2.Sin().Result));

            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var resultMagnitude = sumX.Then(PradOp.SquareOp)
                                      .Then(PradOp.AddOp, sumY.Then(PradOp.SquareOp).Result)
                                      .Then(PradOp.SquareRootOp);

            var resultAngle = sumY.Then(PradOp.Atan2Op, sumX.Result);

            var output = resultMagnitude.Then(PradOp.ConcatOp, new Tensor[] { resultAngle.Result }, axis: 1).Result.ToMatrix();
        }

        [Fact]
        public void TestVNNOperationBack()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)i).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (i * 2) + rand.NextDouble()).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());
            var gradientOfLoss = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => rand.NextDouble()).ToArray());

            ElementwiseVectorConstituentMultiplyOperation op = new ElementwiseVectorConstituentMultiplyOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();
            var backResult = op.Backward(gradientOfLoss.ToMatrix());
            var gradientOfInput1 = (backResult.Item1 as Matrix)!.ToTensor();
            var gradientOfInput2 = (backResult.Item2 as Matrix)!.ToTensor();
            var gradientOfWeights = (backResult.Item3 as Matrix)!.ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            var anglesSeed = opInput1.Indexer(":", $"{halfCols}:");
            var anglesOther = opInput2.Indexer(":", $"{halfCols}:");

            var concatAngles = opInput1.Concat(new[] { anglesOther.Result }, axis: 1);

            var flatAngles = opInput1.Reshape(new int[] { 1, -1 });
            
            var (sinAngles, cosAngles) = opInput1.DoParallel(
                a => a.Sin(),
                a => a.Cos());

            var add = sinAngles.Then(PradOp.AddOp, cosAngles.Result);

            var dInput1 = opInput1.Back(new Tensor(new int[] { 1, 18 }, Enumerable.Range(0, 18).Select(x => rand.NextDouble()).ToArray()));
        }

        [Fact]
        public void TestExpAndMean()
        {
            Tensor tensor = new Tensor(new int[] { 200, 300, 400 }, 5f);

            PradOp op = new PradOp(tensor);

            var results = op.DoParallel(
                x => x.Exp(),
                y => y.Mean(0),
                z => z.Sin()
                );

            var concatResult = results[0].Then(PradOp.ConcatOp, new Tensor[] { results[1].Result, results[2].Result }, 1)
                .Then(PradOp.LogOp)
                .Then(PradOp.SinOp)
                .Then(PradOp.CosOp);

            Tensor grad = new Tensor(new int[] { 200, 900, 400 }, 0.1f);

            var gradient = op.Back(grad);
        }

        [Fact]
        public void TestThenGeneric()
        {
            Tensor tensor = new Tensor(new int[] { 200, 300, 400 }, 5f);

            PradOp op = new PradOp(tensor);

            Matrix m1 = new Matrix(1, 4);

            Matrix m2 = new Matrix(2, 3);

            op.SeedResult
               .Then(new AmplifiedSigmoidOperation())
               .Then(new MatrixMultiplyOperation(), m1)
               .Then(new RMAD.LossOps.MeanSquaredErrorLossOperation(), m2);
        }

        [Fact]
        public void MultiPradOpBranchConcatTest()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => i / 100d).ToArray());
            var input2 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => (i * 2) / 100d).ToArray());
            var weights = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            PradOp input1Op = new PradOp(input1);
            PradOp input2Op = new PradOp(input2);
            PradOp weightsOp = new PradOp(weights);

            var sres1 = input1Op.Square();
            var sres2 = input2Op.Square();
            var sres3 = weightsOp.Square();

            var sres1Branch = sres1.Branch();
            var sres2Branch = sres2.Branch();
            var sres3Branch = sres3.Branch();

            var sres2Sq = sres2.Then(PradOp.SquareOp);
            var sres3Sq = sres3.Then(PradOp.SquareOp);

            var concat = sres1.Then(PradOp.ConcatOp, new Tensor[] { sres2Sq.Result, sres3Sq.Result }, axis: 1);

            var upstreamGradient = new Tensor(new int[] { 3, 60 }, Enumerable.Repeat(1.0, 180).ToArray());

            var backResult = input1Op.Back(upstreamGradient);

            Assert.NotNull(backResult);
            Assert.NotNull(input1Op.SeedGradient);
            Assert.Equal(input1Op.SeedGradient.Shape, input1.Shape);
            Assert.NotNull(input2Op.SeedGradient);
            Assert.Equal(input2Op.SeedGradient.Shape, input2.Shape);
            Assert.NotNull(weightsOp.SeedGradient);
            Assert.Equal(weightsOp.SeedGradient.Shape, weights.Shape);
        }

        [Fact]
        public void MultiPradOpBranchTest()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => i / 100d).ToArray());
            var input2 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => (i * 2) / 100d).ToArray());
            var weights = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            PradOp input1Op = new PradOp(input1);
            PradOp input2Op = new PradOp(input2);
            PradOp weightsOp = new PradOp(weights);

            var sres1 = input1Op.Square();
            var sres2 = input2Op.Square();
            var sres3 = weightsOp.Square();

            var sres1Branch = sres1.Branch();
            var sres2Branch = sres2.Branch();
            var sres3Branch = sres3.Branch();

            var sres2Sq = sres2.Then(PradOp.SquareOp);
            var sres3Sq = sres3.Then(PradOp.SquareOp);

            var concat = sres1.Then(PradOp.ConcatOp, new Tensor[] { sres2Sq.Result, sres3Sq.Result }, axis: 1);

            var addition = sres1Branch.Add(sres2Branch.Result!)
                                .Then(PradOp.AddOp, sres3Branch.Result!);

            var tiledHorizontally = addition.Then(PradOp.TileOp, new int[] { 1, 3 });

            var sum = concat.Then(PradOp.AddOp, tiledHorizontally.Result);

            var upstreamGradient = new Tensor(new int[] { 3, 60 }, Enumerable.Repeat(1.0, 180).ToArray());

            var backResult = input1Op.Back(upstreamGradient);

            Assert.NotNull(backResult);
            Assert.NotNull(input1Op.SeedGradient);
            Assert.Equal(input1Op.SeedGradient.Shape, input1.Shape);
            Assert.NotNull(input2Op.SeedGradient);
            Assert.Equal(input2Op.SeedGradient.Shape, input2.Shape);
            Assert.NotNull(weightsOp.SeedGradient);
            Assert.Equal(weightsOp.SeedGradient.Shape, weights.Shape);

        }

        [Fact]
        public void TestOperatorsOnPradResults()
        {
            var input1 = new Tensor(new int[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => i / 100d).ToArray());
            var input2 = new Tensor(new int[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => i / 100d).ToArray());
            var upstream = new Tensor(new int[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => i / 100d).ToArray());

            var pradOp1 = new PradOp(input1);
            var pradOp2 = new PradOp(input2);

            var sinOp = pradOp1.Sin();
            var cosOp = pradOp2.Cos();

            var multiplied = sinOp * cosOp;

            var addition = sinOp + cosOp;

            var division = sinOp / (PradMath.Atan2(multiplied, addition));

            var subtraction = division - PradMath.Cos(addition);

            var gradient = pradOp1.Back(upstream);
        }

        [Fact]
        public void TestQuotientRule()
        {
            var input1 = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 100d).ToArray());
            var input2 = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 100d).ToArray());
            var weights = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 100d).ToArray());

            var upstream = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 1000d).ToArray());

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);
            var branch = opInput1.Branch();
            var seedGrad = opInput1.SeedGradient;
            var sinRes = opInput1.Sin();
            var sinBranch = sinRes.PradOp.Branch();
            var cosRes = branch.Cos();
            var cosBranch = cosRes.PradOp.Branch();
            var res = sinRes.PradOp.Div(cosRes.Result).PradOp.Mul(opWeights.SeedResult.Result);
            var atanRes = sinBranch.Atan2(cosBranch.BranchInitialTensor);
            var concat = atanRes.PradOp.Indexer("...", ":60").PradOp.Concat(new Tensor[] { opInput2.Indexer("...", "60:").Result }, 1);
            var divRes = res.PradOp.Div(concat.Result);

            var gradRes = divRes.Back(upstream);
        }

        [Fact]
        public void TestMultipleBack()
        {
            var weights1 = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 100d).ToArray());
            var weights2 = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 100d).ToArray());

            var random = new Random(3);

            for (int i = 0; i < 5; ++i)
            {
                var input1 = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) * random.NextDouble() / 100d).ToArray());
                var predicted = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) * random.NextDouble() / 100d).ToArray());
                this.RunOnce(ref input1, ref weights1, ref weights2, ref predicted);
            }
        }

        public void RunOnce(ref Tensor input1, ref Tensor weights1, ref Tensor weights2, ref Tensor predicted)
        {
            var defaultLearningRate = 0.001d;

            PradOp input1Op = new PradOp(input1);
            PradOp weights1Op = new PradOp(weights1);
            PradOp weights2Op = new PradOp(weights2);

            var res = input1Op.Cos()
                .PradOp.Sin()
                .PradOp.Square().PradOp.Mul(weights1Op.BranchInitialTensor)
                .PradOp.Square().PradOp.Mul(weights2Op.BranchInitialTensor)
                .PradOp.Atan2(weights1Op.BranchInitialTensor)
                .PradOp.Atan2(weights2Op.BranchInitialTensor);

            (var lossTensor, var upstream) = res.Result.MeanSquaredError(predicted);

            res.Back(upstream);

            weights1Op.Optimize(PradOptimizer.CreateAdamOptimizer(learningRate: defaultLearningRate));
            weights2Op.Optimize(PradOptimizer.CreateRMSPropOptimizer(learningRate: 0.0001d));
        }

        [Fact]
        public void TestSineSoftmax()
        {
            var input1 = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 100d).ToArray());

            SineSoftmaxOperation op = new SineSoftmaxOperation();
            var forwardRes = op.Forward(input1.ToMatrix());
            var forwardTensor = forwardRes.ToTensor();

            var upstream = new Tensor(new int[] { 6, 120 }, Enumerable.Range(0, 720).Select(i => (i + 1) / 1000d).ToArray());

            var result = op.Backward(upstream.ToMatrix());
            var resultMatrix = result.Item1 as Matrix;
            var resultTensor = resultMatrix.ToTensor();

            var result2 = op.SineSoftmaxBackward(input1.ToMatrix(), upstream.ToMatrix());
            var result2Tensor = result2.ToTensor();

            var opInput1 = new PradOp(input1);
            var sinned = opInput1.Sin();
            var exped = sinned.Then(PradOp.ExpOp);

            // Determine the axis to sum over based on the input dimensions
            int sumAxis = opInput1.CurrentShape.Length - 1;  // Last dimension

            var expedBranch = exped.BranchStack(2);
            var sums = exped.PradOp.Sum(new[] { sumAxis });
            var broadcastedSums = sums.Then(PradOp.BroadcastToOp, input1.Shape);
            var denominator = broadcastedSums.PradOp.Add(expedBranch.Pop().BranchInitialTensor);
            var output = denominator.PradOp.DivInto(expedBranch.Pop().BranchInitialTensor);

            var gradientOutput = output.Back(upstream);
            var gradientInput = opInput1.SeedGradient;
        }

        [Fact]
        public void TestInterleavedGather()
        {
            var input1 = new Tensor(new int[] { 3, 6, 12 }, Enumerable.Range(0, 216).Select(i => (i + 1) / 100d).ToArray());
            var opInput1 = new PradOp(input1);
            var opInput1Branch = opInput1.Branch();

            var interleavedResult = opInput1.InterleavedGather(6, 12);
            var tensor = interleavedResult.Result;

            var inverse = interleavedResult.PradOp.InterleavedGatherInverse(6, 12);
            var inverseTensor = inverse.Result;
        }

        [Fact]
        public void TestVectorConvolution()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 1, 3, 6, 6 }, Enumerable.Range(0, 108).Select(i => (i + 1) / 100d).ToArray());
            var opInput1 = new PradOp(input1);
            var opInputBranch = opInput1.Branch();

            var magnitudes = opInput1.Indexer("...", ":3");
            var angles = opInputBranch.Indexer("...", "3:");

            var magPatches = magnitudes.PradOp.ExtractPatches(new int[] { 3, 3 }, new int[] { 1, 1 }, "SAME");

            var anglePatches = angles.PradOp.ExtractPatches(new int[] { 3, 3 }, new int[] { 1, 1 }, "SAME");

            var magPatchesShapeLast = magPatches.PradOp.CurrentShape[^1];

            var anglePatchesShapeLast = anglePatches.PradOp.CurrentShape[^1];

            var magFlattened = magPatches.PradOp.Reshape(new int[] { 1, -1, magPatchesShapeLast });

            var angleFlattened = anglePatches.PradOp.Reshape(new int[] { 1, -1, anglePatchesShapeLast });

            var combinedPatches = magFlattened.PradOp.Concat(new Tensor[] { angleFlattened.Result }, -1);

            // TODO: Perform convolution
        }

        [Fact]
        public void TestVNNDecompositionReverse()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => (i+1) / 100d).ToArray());
            var input2 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => ((i+1) * 2) / 100d).ToArray());
            var weights = new Tensor(new int[] { 3, 2 }, Enumerable.Range(0, 6).Select(i => PradMath.Pow(rand.NextDouble(), 2d)).ToArray());

            ElementwiseVectorDecompositionOperation op = new ElementwiseVectorDecompositionOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            //            num_rows, num_cols = tf.shape(input1)[1], tf.shape(input1)[2] // 2
            //            size = num_rows * num_cols

            var num_rows = input1.Shape[0];
            var num_cols = input1.Shape[1] / 2;
            var size = num_rows * num_cols;

            //        # Split input1 into magnitude and angle
            //        magnitude = input1[:, :, :num_cols]
            //        angle = input1[:, :, num_cols:]

            var (magnitude, angle) = opInput1.DoParallel(
                x => x.Indexer(":", $":{num_cols}"),
                y => y.Indexer(":", $"{num_cols}:"));
            var magnitudeBranch = magnitude.Branch();
            var angleBranch = angle.Branch();

            //        # Extract components from input2
            //        input2_cols = tf.shape(input2)[2]
            //        half_cols = input2_cols // 2

            var input2_cols = input2.Shape[1];
            var half_cols = input2_cols / 2;

            //        # Extract other components
            //        w_magnitudes = tf.stack([input2[:, :, 1 + i:half_cols: 5] for i in range(4)], axis = -1)
            //                w_angles = tf.stack([input2[:, :, half_cols + 1 + i::5] for i in range(4)], axis = -1)

            var opInput2Branch = opInput2.Branch();

            var opInput2Branches = opInput2.BranchStack(3);

            var w_magnitudes_t = new PradResult[4];
            var w_angles_t = new PradResult[4];
            for (int i = 0; i < 4; i++)
            {
                var branchM = opInput2;

                if (i > 0)
                {
                    branchM = opInput2Branches.Pop();
                }

                var (w_magnitudes_tt, w_angles_tt) = branchM.DoParallel(
                    x => x.Indexer(":", $"{1 + i}:{half_cols}:5"),
                    y => y.Indexer(":", $"{half_cols + 1 + i}::5")
                    );

                w_magnitudes_t[i] = w_magnitudes_tt;
                w_angles_t[i] = w_angles_tt;
            }

            var w_magnitudes_stacked = w_magnitudes_t[0].PradOp.Stack(w_magnitudes_t.Select(f => f.Result).Skip(1).ToArray(), axis: -1);
            var w_angles_stacked = w_angles_t[0].PradOp.Stack(w_angles_t.Select(f => f.Result).Skip(1).ToArray(), axis: -1);

            var w_magnitudes = w_magnitudes_stacked;
            var w_angles = w_angles_stacked;
            var w_magnitudesBranch = w_magnitudes.Branch();
            var w_anglesBranch = w_angles.Branch();

            // # Reshape and transpose operations
            // w_magnitudes_trans = tf.transpose(tf.reshape(w_magnitudes, [1, 4, -1]), [0, 2, 1])
            // w_angles_trans = tf.transpose(tf.reshape(w_angles, [1, 4, -1]), [0, 2, 1])

            var w_magnitudesTrans = w_magnitudesBranch.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });
            var w_anglesTrans = w_anglesBranch.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            //        # Correctly extract w_magnitude_pivot and w_angle_pivot
            //        w_magnitude_pivot = input2[:, :, :half_cols][:, :, ::5]
            //        w_angle_pivot = input2[:, :, half_cols:][:, :, ::5]

            var (w_magnitude_pivot_a, w_angle_pivot_a) = opInput2Branch.DoParallel(
                x => x.Indexer(":", $":{half_cols}"),
                y => y.Indexer(":", $"{half_cols}:")
            );

            var w_magnitude_pivot = w_magnitude_pivot_a.PradOp.Indexer(":", $"::5");
            var w_angle_pivot = w_angle_pivot_a.PradOp.Indexer(":", $"::5");

            var w_magnitude_pivotBranch = w_magnitude_pivot.Branch();
            var w_angle_pivotBranch = w_angle_pivot.Branch();

            //# Compute x and y components
            //                x = magnitude * tf.math.cos(angle)
            //        y = magnitude * tf.math.sin(angle)

            var (cosAngles, sinAngles) = angle.PradOp.DoParallel(
                x => x.Cos(),
                y => y.Sin()
                );

            var (x, y) = magnitude.PradOp.DoParallel(
                x => x.Mul(cosAngles.Result),
                x => x.Mul(sinAngles.Result));

            var xResult = x.Result;
            var yResult = y.Result;

            //        x_pivot = w_magnitude_pivot * tf.math.cos(w_angle_pivot)
            //        y_pivot = w_magnitude_pivot * tf.math.sin(w_angle_pivot)

            var (cosPivot, sinPivot) = w_angle_pivot.PradOp.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (xPivot, yPivot) = w_magnitude_pivot.PradOp.DoParallel(
                x => x.Mul(cosPivot.Result),
                x => x.Mul(sinPivot.Result));

            var xPivotResult = xPivot.Result;
            var yPivotResult = yPivot.Result;

            //        x_w = w_magnitudes * tf.math.cos(w_angles)
            //        y_w = w_magnitudes * tf.math.sin(w_angles)

            var (cosAngles_w, sinAngles_w) = w_angles.PradOp.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (x_w, y_w) = w_magnitudes.PradOp.DoParallel(
                x => x.Mul(cosAngles_w.Result),
                x => x.Mul(sinAngles_w.Result));

            //        # Adjust weights
            //        weights = tf.add(weights, 0.01)

            Tensor addScalar = new Tensor(weights.Shape, 0.01d);
            var adjustedWeights = opWeights.Add(addScalar);

            //        # Compute sum components
            //        sum_x = (x + x_pivot) / (weights + 1e-9)
            //        sum_y = (y + y_pivot) / (weights + 1e-9)

            var xPivotAdd = x.Then(PradOp.AddOp, xPivot.Result);
            var yPivotAdd = y.Then(PradOp.AddOp, yPivot.Result);

            var weightsEpsilon = adjustedWeights.Then(PradOp.AddOp, new Tensor(weights.Shape, 1e-9d));

            var sumX = xPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);
            var sumY = yPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);

            //        # Compute differences
            //        sum_x_expanded = tf.expand_dims(sum_x, -1)
            //        sum_y_expanded = tf.expand_dims(sum_y, -1)

            var sumXExpanded = sumX.Then(PradOp.ExpandDimsOp, -1);
            var sumYExpanded = sumY.Then(PradOp.ExpandDimsOp, -1);

            var sumXBranch = sumX.Branch();
            var negativeSumXExpanded = sumXBranch.Mul(new Tensor(sumXBranch.CurrentShape, -1d));

            var sumYBranch = sumY.Branch();
            var negativeSumYExpanded = sumYBranch.Mul(new Tensor(sumYBranch.CurrentShape, -1d));

            // # Vectorized difference calculation
            // sum_x_reshaped = tf.reshape(sum_x_expanded, [1, -1])
            // negative_sum_x_reshaped = tf.reshape(-sum_x_expanded, [1, -1])
            // x_w_reshaped = tf.reshape(x_w, [4, -1])
            // sum_x_concatenated = tf.concat([sum_x_reshaped, -sum_x_reshaped, sum_x_reshaped, -sum_x_reshaped], axis=0)
            // diff_x = sum_x_concatenated - x_w_reshaped


            // Reshape sumXExpanded and negativeSumXExpanded
            var sumXReshaped = sumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumXReshaped = negativeSumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            // Reshape x_w
            var x_wReshaped = x_w.Then(PradOp.ReshapeOp, new int[] { 4, size });

            var sumXReshapedBranch = sumXReshaped.Branch();

            // Concatenate sumX and negativeSumX horizontally
            var sumXConcatenated = sumXReshaped.Then(PradOp.ConcatOp,
                new[] { negativeSumXReshaped.Result, sumXReshapedBranch.BranchInitialTensor, negativeSumXReshaped.Result },
                axis: 0);

            // Perform the subtraction
            var diffX = x_wReshaped.PradOp.SubFrom(sumXConcatenated.Result);


            // # Vectorized difference calculation
            // sum_y_reshaped = tf.reshape(sum_y_expanded, [1, -1])
            // negative_sum_y_reshaped = tf.reshape(-sum_y_expanded, [1, -1])
            // y_w_reshaped = tf.reshape(y_w, [4, -1])
            // sum_y_concatenated = tf.concat([sum_y_reshaped, -sum_y_reshaped, sum_y_reshaped, -sum_y_reshaped], axis=0)
            // diff_y = sum_y_concatenated - y_w_reshaped

            // Repeat the process for Y
            var sumYReshaped = sumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumYReshaped = negativeSumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            var y_wReshaped = y_w.Then(PradOp.ReshapeOp, new int[] { 4, size });

            var sumYReshapedBranch = sumYReshaped.Branch();

            var sumYConcatenated = sumYReshaped.Then(PradOp.ConcatOp,
                new[] { negativeSumYReshaped.Result, sumYReshapedBranch.BranchInitialTensor, negativeSumYReshaped.Result },
                axis: 0);

            var diffY = sumYConcatenated.Then(PradOp.SubOp, y_wReshaped.Result);

            Debug.WriteLine($"PRAD diffX: {diffX.Result.PrintCode(8)}");
            Debug.WriteLine($"PRAD diffY: {diffY.Result.PrintCode(8)}");

            var diffXBranch = diffX.Branch();
            var diffYBranch = diffY.Branch();

            //        # Compute result magnitudes and angles
            //        result_magnitudes = tf.math.sqrt(diff_x * *2 + diff_y * *2)
            //        result_angles = tf.math.atan2(diff_y, diff_x)

            var diffXSquared = diffX.Then(PradOp.SquareOp);
            var diffYSquared = diffY.Then(PradOp.SquareOp);

            var resultMagnitudes = diffXSquared.Then(PradOp.AddOp, diffYSquared.Result)
                                                .Then(PradOp.SquareRootOp);

            var resultAngles = diffYBranch.Atan2(diffXBranch.BranchInitialTensor);

            // Result debugging
            Debug.WriteLine($"PRAD resultMagnitudes: {resultMagnitudes.Result.PrintCode(8)}");
            Debug.WriteLine($"PRAD resultAngles: {resultAngles.Result.PrintCode(8)}");


            var naiveOutputCode2 = resultTensor.PrintCode();

            // # Reshape and transpose results
            // result_magnitudes_trans = tf.transpose(tf.reshape(result_magnitudes, [1, 4, -1]), [0, 2, 1])
            // result_angles_trans = tf.transpose(tf.reshape(result_angles, [1, 4, -1]), [0, 2, 1])

            // reshaped_result_magnitudes = tf.reshape(result_magnitudes_trans, [num_rows, -1])
            // reshaped_result_angles = tf.reshape(result_angles_trans, [num_rows, -1])

            var resultMagnitudesTrans2 = resultMagnitudes.PradOp.Reshape(1, 4, -1);
            var resultMagnitudesTrans = resultMagnitudesTrans2.PradOp.Transpose(new int[] { 0, 2, 1 });
            var resultAnglesTrans = resultAngles.PradOp.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            var reshapedResultMagnitudes = resultMagnitudesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            var reshapedResultAngles = resultAnglesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            // # Final output construction
            // magnitude_reshaped = tf.reshape(magnitude, [3, 2, 1])
            // angle_reshaped = tf.reshape(angle, [3, 2, 1])
            // w_magnitude_pivot_reshaped = tf.reshape(w_magnitude_pivot, [3, 2, 1])
            // w_angle_pivot_reshaped = tf.reshape(w_angle_pivot, [3, 2, 1])
            // w_magnitudes_reshaped = tf.reshape(w_magnitudes_trans, [3, 2, 4])
            // w_angles_reshaped = tf.reshape(w_angles_trans, [3, 2, 4])
            // result_magnitudes_reshaped = tf.reshape(reshaped_result_magnitudes, [3, 2, 4])
            // result_angles_reshaped = tf.reshape(reshaped_result_angles, [3, 2, 4])

            // First, let's reshape our tensors to make them easier to work with
            var magnitudeReshaped = magnitudeBranch.Reshape(new int[] { 3, 2, 1 });
            var angleReshaped = angleBranch.Reshape(new int[] { 3, 2, 1 });
            var w_magnitude_pivotReshaped = w_magnitude_pivotBranch.Reshape(new int[] { 3, 2, 1 });
            var w_angle_pivotReshaped = w_angle_pivotBranch.Reshape(new int[] { 3, 2, 1 });
            var w_magnitudesReshaped = w_magnitudesTrans.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });
            var w_anglesReshaped = w_anglesTrans.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });
            var resultMagnitudesReshaped = reshapedResultMagnitudes.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });
            var resultAnglesReshaped = reshapedResultAngles.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });

            // Now, let's create the interleaved structure for magnitudes and angles separately
            var magnitudesPart = resultMagnitudesReshaped.Then(PradOp.ConcatOp, new[] {
                magnitudeReshaped.Result,
                w_magnitude_pivotReshaped.Result,
                w_magnitudesReshaped.Result,
            }, axis: 2, new int[] { 1, 2, 3, 0 });

            // # Concatenate all parts
            // magnitudes_part = tf.concat([magnitude_reshaped, w_magnitude_pivot_reshaped, w_magnitudes_reshaped, result_magnitudes_reshaped], axis=2)
            // angles_part = tf.concat([angle_reshaped, w_angle_pivot_reshaped, w_angles_reshaped, result_angles_reshaped], axis=2)

            var anglesPart = angleReshaped.Then(PradOp.ConcatOp, new[] {
                w_angle_pivotReshaped.Result,
                w_anglesReshaped.Result,
                resultAnglesReshaped.Result
            }, axis: 2);

            // # Combine magnitudes and angles
            // output = tf.concat([magnitudes_part, angles_part], axis=1)
            // final_output = tf.reshape(output, [3, 40])

            // Combine magnitudes and angles
            var output = magnitudesPart.Then(PradOp.ConcatOp, new[] { anglesPart.Result }, axis: 1);

            // Reshape to final output shape
            var finalOutput = output.Then(PradOp.ReshapeOp, new int[] { 3, 40 });

            var pradOpOutputCode = finalOutput.Result.PrintCode();
            var naiveOutputCode = resultTensor.PrintCode();

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            var upstream = finalOutput.Result.DeepClone();

            finalOutput.Back(upstream);
            var o1 = opInput1.SeedGradient;
            var o2 = opInput2.SeedGradient;
            var ow = opWeights.SeedGradient;

        }

        [Fact]
        public void TestVNNDecompositionOperationUsingIndexer()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => i / 100d).ToArray());
            var input2 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => (i * 2) / 100d).ToArray());
            var weights = new Tensor(new int[] { 3, 2 }, Enumerable.Range(0, 6).Select(i => PradMath.Pow(rand.NextDouble(), 2d)).ToArray());

            ElementwiseVectorDecompositionOperation op = new ElementwiseVectorDecompositionOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            //            num_rows, num_cols = tf.shape(input1)[1], tf.shape(input1)[2] // 2
            //            size = num_rows * num_cols

            var num_rows = input1.Shape[0];
            var num_cols = input1.Shape[1] / 2;
            var size = num_rows * num_cols;

            //        # Split input1 into magnitude and angle
            //        magnitude = input1[:, :, :num_cols]
            //        angle = input1[:, :, num_cols:]

            var (magnitude, angle) = opInput1.DoParallel(
                x => x.Indexer(":", $":{num_cols}"),
                y => y.Indexer(":", $"{num_cols}:"));
            var magnitudeBranch = magnitude.Branch();
            var angleBranch = angle.Branch();

            //        # Extract components from input2
            //        input2_cols = tf.shape(input2)[2]
            //        half_cols = input2_cols // 2

            var input2_cols = input2.Shape[1];
            var half_cols = input2_cols / 2;

            //        # Extract other components
            //        w_magnitudes = tf.stack([input2[:, :, 1 + i:half_cols: 5] for i in range(4)], axis = -1)
            //                w_angles = tf.stack([input2[:, :, half_cols + 1 + i::5] for i in range(4)], axis = -1)

            var opInput2Branch = opInput2.Branch();

            var w_magnitudes_t = new Tensor[4];
            var w_angles_t = new Tensor[4];
            for (int i = 0; i < 4; i++)
            {
                var branchM = opInput2.Branch();

                var(w_magnitudes_tt, w_angles_tt) = branchM.DoParallel(
                    x => x.Indexer(":", $"{1 + i}:{half_cols}:5"),
                    y => y.Indexer(":", $"{half_cols + 1 + i}::5")
                    );

                w_magnitudes_t[i] = w_magnitudes_tt.Result;
                w_angles_t[i] = w_angles_tt.Result;
            }

            var w_magnitudes_stacked = new PradOp(w_magnitudes_t[0]).Stack(w_magnitudes_t.Skip(1).ToArray(), axis: -1);
            var w_angles_stacked = new PradOp(w_angles_t[0]).Stack(w_angles_t.Skip(1).ToArray(), axis: -1);

            var w_magnitudes = w_magnitudes_stacked;
            var w_angles = w_angles_stacked;
            var w_magnitudesBranch = w_magnitudes.Branch();
            var w_anglesBranch = w_angles.Branch();

            // # Reshape and transpose operations
            // w_magnitudes_trans = tf.transpose(tf.reshape(w_magnitudes, [1, 4, -1]), [0, 2, 1])
            // w_angles_trans = tf.transpose(tf.reshape(w_angles, [1, 4, -1]), [0, 2, 1])

            var w_magnitudesTrans = w_magnitudesBranch.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });
            var w_anglesTrans = w_anglesBranch.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            //        # Correctly extract w_magnitude_pivot and w_angle_pivot
            //        w_magnitude_pivot = input2[:, :, :half_cols][:, :, ::5]
            //        w_angle_pivot = input2[:, :, half_cols:][:, :, ::5]

            var (w_magnitude_pivot, w_angle_pivot) = opInput2Branch.DoParallel(
                x => x.Indexer(":", $":{half_cols}").PradOp.Indexer(":", $"::5"),
                y => y.Indexer(":", $"{half_cols}:").PradOp.Indexer(":", $"::5")
            );
            var w_magnitude_pivotBranch = w_magnitude_pivot.Branch();
            var w_angle_pivotBranch = w_angle_pivot.Branch();

            //# Compute x and y components
            //                x = magnitude * tf.math.cos(angle)
            //        y = magnitude * tf.math.sin(angle)

            var (cosAngles, sinAngles) = angle.PradOp.DoParallel(
                x => x.Cos(),
                y => y.Sin()
                );

            var (x, y) = magnitude.PradOp.DoParallel(
                x => x.Mul(cosAngles.Result), 
                x => x.Mul(sinAngles.Result));

            var xResult = x.Result;
            var yResult = y.Result;

            //        x_pivot = w_magnitude_pivot * tf.math.cos(w_angle_pivot)
            //        y_pivot = w_magnitude_pivot * tf.math.sin(w_angle_pivot)

            var (cosPivot, sinPivot) = w_angle_pivot.PradOp.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (xPivot, yPivot) = w_magnitude_pivot.PradOp.DoParallel(
                x => x.Mul(cosPivot.Result),
                x => x.Mul(sinPivot.Result));

            var xPivotResult = xPivot.Result;
            var yPivotResult = yPivot.Result;

            //        x_w = w_magnitudes * tf.math.cos(w_angles)
            //        y_w = w_magnitudes * tf.math.sin(w_angles)

            var (cosAngles_w, sinAngles_w) = w_angles.PradOp.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (x_w, y_w) = w_magnitudes.PradOp.DoParallel(
                x => x.Mul(cosAngles_w.Result),
                x => x.Mul(sinAngles_w.Result));

            //        # Adjust weights
            //        weights = tf.add(weights, 0.01)

            Tensor addScalar = new Tensor(weights.Shape, 0.01d);
            var adjustedWeights = opWeights.Add(addScalar);

            //        # Compute sum components
            //        sum_x = (x + x_pivot) / (weights + 1e-9)
            //        sum_y = (y + y_pivot) / (weights + 1e-9)

            var xPivotAdd = x.Then(PradOp.AddOp, xPivot.Result);
            var yPivotAdd = y.Then(PradOp.AddOp, yPivot.Result);

            var weightsEpsilon = adjustedWeights.Then(PradOp.AddOp, new Tensor(weights.Shape, 1e-9d));

            var sumX = xPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);
            var sumY = yPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);

            //        # Compute differences
            //        sum_x_expanded = tf.expand_dims(sum_x, -1)
            //        sum_y_expanded = tf.expand_dims(sum_y, -1)

            var sumXExpanded = sumX.Then(PradOp.ExpandDimsOp, -1);
            var sumYExpanded = sumY.Then(PradOp.ExpandDimsOp, -1);

            var sumXBranch = sumX.Branch();
            var negativeSumXExpanded = sumXBranch.Mul(new Tensor(sumXBranch.CurrentShape, -1d));

            var sumYBranch = sumY.Branch();
            var negativeSumYExpanded = sumYBranch.Mul(new Tensor(sumYBranch.CurrentShape, -1d));

            // # Vectorized difference calculation
            // sum_x_reshaped = tf.reshape(sum_x_expanded, [1, -1])
            // negative_sum_x_reshaped = tf.reshape(-sum_x_expanded, [1, -1])
            // x_w_reshaped = tf.reshape(x_w, [4, -1])
            // sum_x_concatenated = tf.concat([sum_x_reshaped, -sum_x_reshaped, sum_x_reshaped, -sum_x_reshaped], axis=0)
            // diff_x = sum_x_concatenated - x_w_reshaped


            // Reshape sumXExpanded and negativeSumXExpanded
            var sumXReshaped = sumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumXReshaped = negativeSumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            // Reshape x_w
            var x_wReshaped = x_w.Then(PradOp.ReshapeOp, new int[] { 4, size });

            // Concatenate sumX and negativeSumX horizontally
            var sumXConcatenated = sumXReshaped.Then(PradOp.ConcatOp,
                new[] { negativeSumXReshaped.Result, sumXReshaped.Result, negativeSumXReshaped.Result },
                axis: 0);

            // Perform the subtraction
            var diffX = sumXConcatenated.Then(PradOp.SubOp, x_wReshaped.Result);


            // # Vectorized difference calculation
            // sum_y_reshaped = tf.reshape(sum_y_expanded, [1, -1])
            // negative_sum_y_reshaped = tf.reshape(-sum_y_expanded, [1, -1])
            // y_w_reshaped = tf.reshape(y_w, [4, -1])
            // sum_y_concatenated = tf.concat([sum_y_reshaped, -sum_y_reshaped, sum_y_reshaped, -sum_y_reshaped], axis=0)
            // diff_y = sum_y_concatenated - y_w_reshaped

            // Repeat the process for Y
            var sumYReshaped = sumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumYReshaped = negativeSumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            var y_wReshaped = y_w.Then(PradOp.ReshapeOp, new int[] { 4, size });

            var sumYConcatenated = sumYReshaped.Then(PradOp.ConcatOp,
                new[] { negativeSumYReshaped.Result, sumYReshaped.Result, negativeSumYReshaped.Result },
                axis: 0);

            var diffY = sumYConcatenated.Then(PradOp.SubOp, y_wReshaped.Result);

            Debug.WriteLine($"PRAD diffX: {diffX.Result.PrintCode(8)}");
            Debug.WriteLine($"PRAD diffY: {diffY.Result.PrintCode(8)}");

            var diffXBranch = diffX.Branch();
            var diffYBranch = diffY.Branch();

            //        # Compute result magnitudes and angles
            //        result_magnitudes = tf.math.sqrt(diff_x * *2 + diff_y * *2)
            //        result_angles = tf.math.atan2(diff_y, diff_x)

            var diffXSquared = diffX.Then(PradOp.SquareOp);
            var diffYSquared = diffY.Then(PradOp.SquareOp);

            var resultMagnitudes = diffXSquared.Then(PradOp.AddOp, diffYSquared.Result)
                                                .Then(PradOp.SquareRootOp);

            var resultAngles = diffYBranch.Atan2(diffXBranch.Result!);

            // Result debugging
            Debug.WriteLine($"PRAD resultMagnitudes: {resultMagnitudes.Result.PrintCode(8)}");
            Debug.WriteLine($"PRAD resultAngles: {resultAngles.Result.PrintCode(8)}");


            var naiveOutputCode2 = resultTensor.PrintCode();

            // # Reshape and transpose results
            // result_magnitudes_trans = tf.transpose(tf.reshape(result_magnitudes, [1, 4, -1]), [0, 2, 1])
            // result_angles_trans = tf.transpose(tf.reshape(result_angles, [1, 4, -1]), [0, 2, 1])

            // reshaped_result_magnitudes = tf.reshape(result_magnitudes_trans, [num_rows, -1])
            // reshaped_result_angles = tf.reshape(result_angles_trans, [num_rows, -1])

            var resultMagnitudesTrans2 = resultMagnitudes.PradOp.Reshape(1, 4, -1);
            var resultMagnitudesTrans = resultMagnitudesTrans2.PradOp.Transpose(new int[] { 0, 2, 1 });
            var resultAnglesTrans = resultAngles.PradOp.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            var reshapedResultMagnitudes = resultMagnitudesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            var reshapedResultAngles = resultAnglesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            // # Final output construction
            // magnitude_reshaped = tf.reshape(magnitude, [3, 2, 1])
            // angle_reshaped = tf.reshape(angle, [3, 2, 1])
            // w_magnitude_pivot_reshaped = tf.reshape(w_magnitude_pivot, [3, 2, 1])
            // w_angle_pivot_reshaped = tf.reshape(w_angle_pivot, [3, 2, 1])
            // w_magnitudes_reshaped = tf.reshape(w_magnitudes_trans, [3, 2, 4])
            // w_angles_reshaped = tf.reshape(w_angles_trans, [3, 2, 4])
            // result_magnitudes_reshaped = tf.reshape(reshaped_result_magnitudes, [3, 2, 4])
            // result_angles_reshaped = tf.reshape(reshaped_result_angles, [3, 2, 4])

            // First, let's reshape our tensors to make them easier to work with
            var magnitudeReshaped = magnitudeBranch.Reshape(new int[] { 3, 2, 1 });
            var angleReshaped = angleBranch.Reshape(new int[] { 3, 2, 1 });
            var w_magnitude_pivotReshaped = w_magnitude_pivotBranch.Reshape(new int[] { 3, 2, 1 });
            var w_angle_pivotReshaped = w_angle_pivotBranch.Reshape(new int[] { 3, 2, 1 });
            var w_magnitudesReshaped = w_magnitudesTrans.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });
            var w_anglesReshaped = w_anglesTrans.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });
            var resultMagnitudesReshaped = reshapedResultMagnitudes.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });
            var resultAnglesReshaped = reshapedResultAngles.Then(PradOp.ReshapeOp, new int[] { 3, 2, 4 });

            // Now, let's create the interleaved structure for magnitudes and angles separately
            var magnitudesPart = magnitudeReshaped.Then(PradOp.ConcatOp, new[] {
                w_magnitude_pivotReshaped.Result,
                w_magnitudesReshaped.Result,
                resultMagnitudesReshaped.Result
            }, axis: 2);

            // # Concatenate all parts
            // magnitudes_part = tf.concat([magnitude_reshaped, w_magnitude_pivot_reshaped, w_magnitudes_reshaped, result_magnitudes_reshaped], axis=2)
            // angles_part = tf.concat([angle_reshaped, w_angle_pivot_reshaped, w_angles_reshaped, result_angles_reshaped], axis=2)

            var anglesPart = angleReshaped.Then(PradOp.ConcatOp, new[] {
                w_angle_pivotReshaped.Result,
                w_anglesReshaped.Result,
                resultAnglesReshaped.Result
            }, axis: 2);

            // # Combine magnitudes and angles
            // output = tf.concat([magnitudes_part, angles_part], axis=1)
            // final_output = tf.reshape(output, [3, 40])

            // Combine magnitudes and angles
            var output = magnitudesPart.Then(PradOp.ConcatOp, new[] { anglesPart.Result }, axis: 1);

            // Reshape to final output shape
            var finalOutput = output.Then(PradOp.ReshapeOp, new int[] { 3, 40 });

            var pradOpOutputCode = finalOutput.Result.PrintCode();
            var naiveOutputCode = resultTensor.PrintCode();

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            Assert.Equal(naiveOutputCode, pradOpOutputCode);
        }

        [Fact]
        public void TestBezierBack()
        {
            var size1 = 1200000;
            var size2 = size1 * 3;
            var size3 = size1 / 2;
            var size4 = size3 * 3;
            Random rand = new Random(3);
            var input1 = new Tensor(new int[] { 3, size1 }, Enumerable.Range(0, size2).Select(i => (double)i).ToArray());
            var input2 = new Tensor(new int[] { 3, size1 }, Enumerable.Range(0, size2).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, size3 }, Enumerable.Range(0, size4).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            // Create Bezier Waveforms
            var N_sin = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.5, size4).ToArray());
            var N_cos = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.5, size4).ToArray());
            var p0 = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(0.5, size4).ToArray());
            var p1 = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.0, size4).ToArray());
            var p2 = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.5, size4).ToArray());

            // Parameters for Atan2 approximation
            var alpha = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.0, size4).ToArray());
            var beta = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.0, size4).ToArray());
            var lambda = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.0, size4).ToArray());
            var gamma = new Tensor(new int[] { 3, size3 }, Enumerable.Repeat(1.0, size4).ToArray());

            // Create PradOp instances for Bezier control points
            var opN_sin = new PradOp(N_sin);
            var opN_cos = new PradOp(N_cos);
            var opP0 = new PradOp(p0);
            var opP1 = new PradOp(p1);
            var opP2 = new PradOp(p2);

            // GradientRecorder.Instance.RecordingEnabled = true;
            Stopwatch sw = new Stopwatch();
            sw.Start();

            // PradOp implementation
            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            // Create branches for magnitudes and angles
            var magnitude1Branch = magnitude1.Branch();
            var magnitude2Branch = magnitude2.Branch();
            var angle1Branch = angle1.Branch();
            var angle2Branch = angle2.Branch();

            var branchedOpNCos = opN_cos.Branch();
            var branchedOpNSin = opN_sin.Branch();
            var bOpNCos = opN_cos.Branch();
            var bOpNSin = opN_sin.Branch();
            var branchedP0 = opP0.Branch();
            var branchedP1 = opP1.Branch();
            var branchedP2 = opP2.Branch();

            var bP0 = opP0.Branch();
            var bP1 = opP1.Branch();
            var bP2 = opP2.Branch();

            var b4P0 = opP0.Branch();
            var b4P1 = opP1.Branch();
            var b4P2 = opP2.Branch();

            var b5P0 = opP0.Branch();
            var b5P1 = opP1.Branch();
            var b5P2 = opP2.Branch();

            // Compute vector components using Bezier waveforms
            var wave1 = BezierWaveform(angle1, opN_cos, opP0, opP1, opP2).Result;
            var x1 = magnitude1.Mul(wave1);
            var wave2 = BezierWaveform(angle1Branch, opN_sin, branchedP0, branchedP1, branchedP2).Result;
            var y1 = magnitude1Branch.Mul(wave2);
            var wave3 = BezierWaveform(angle2, branchedOpNCos, bP0, bP1, bP2).Result;
            var x2 = magnitude2.Mul(wave3);
            var wave4 = BezierWaveform(angle2Branch, branchedOpNSin, b4P0, b4P1, b4P2).Result;
            var y2 = magnitude2Branch.Mul(wave4);

            // Sum components
            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            // Compute resultant vector magnitude and angle
            var sumXSquared = sumX.PradOp.Square();
            var sumYSquared = sumY.PradOp.Square();
            var magnitudeSquared = sumXSquared.Then(PradOp.AddOp, sumYSquared.Result);
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp)
                                                  .Then(PradOp.MulOp, opWeights.SeedResult.Result);

            // Approximate Atan2 using Bezier waveforms
            var opAlpha = new PradOp(alpha);
            var opBeta = new PradOp(beta);
            var opLambda = new PradOp(lambda);
            var opGamma = new PradOp(gamma);

            var resultAngle = Atan2Approximation(sumYBranch, sumXBranch,
                                                 opAlpha, opBeta, opLambda, opGamma,
                                                 bOpNCos, bOpNSin,
                                                 b5P0, b5P1, b5P2);

            // Concatenate results
            var res = resultMagnitude.PradOp.Concat(new[] { resultAngle.Result }, axis: 1);

            sw.Stop();
            var millis = sw.ElapsedMilliseconds;

            var gradient = new Tensor(new int[] { 3, size1 }, Enumerable.Range(0, size2).Select(i => 1.0).ToArray());
            
            // res.Back(gradient);
            // var steps = GradientRecorder.Instance.GetRecordedGradients();
            
        }

        public PradResult BezierWaveform(PradOp x, PradOp N, PradOp p0, PradOp p1, PradOp p2)
        {
            // Calculate interval limits based on N
            var nSquared = N.Square();
            var nSquaredBranch = nSquared.Branch();
            var halfNSquared = nSquared.Then(PradOp.MulOp, new Tensor(N.CurrentShape, 0.5));

            // Determine the segment and calculate the relative position
            var xMod = x.Modulus(nSquaredBranch.BranchInitialTensor);
            var xModBranch = xMod.Branch();
            var segment = xModBranch.LessThan(halfNSquared.Result);
            var t = xMod.PradOp.Div(halfNSquared.Result);
            var tMod = t.PradOp.Modulus(new Tensor(t.PradOp.CurrentShape, 1.0));

            var branchedTMod = tMod.Branch();
            // Compute Bezier curve for the first segment
            var branchedP0 = p0.Branch();
            var branchedP1 = p1.Branch();
            var branchedP2 = p2.Branch();

            var y1 = CubicBezier(tMod.PradOp, p0.SeedResult, p1.SeedResult, p2.SeedResult);
            // Reflect the Bezier curve for the second segment
            var y2 = CubicBezier(branchedTMod, branchedP0.Mul(new Tensor(p0.CurrentShape, -1.0)),
                branchedP1.Mul(new Tensor(p1.CurrentShape, -1.0)),
                branchedP2.Mul(new Tensor(p2.CurrentShape, -1.0)));

            // Choose between the segments
            var waveformResult = y1.Then(PradOp.WhereOp, segment.Result, y2.Result);
            return waveformResult;
        }

        public PradResult CubicBezier(PradOp t, PradResult p0, PradResult p1, PradResult p2)
        {
            /*
             double t2 = t * t;
            double t3 = t2 * t;
            double t4 = t3 * t;

            double mt = 1.0 - t;
            double mt2 = mt * mt;
            double mt3 = mt2 * mt;

            // Calculate the contribution from each control point
            var r0 = (4 * mt3 * t) * p0;
            var r1 = (6 * mt2 * t2) * p1;
            var r2 = (4 * mt * t3) * p2;
             */

            var tBranches = t.BranchStack(4);
            var t2 = t.Square();
            var t2Branch = t2.Branch();
            var t3 = t2.Then(PradOp.MulOp, tBranches.Pop().BranchInitialTensor);

            var mt = tBranches.Pop().SubFrom(new Tensor(t.CurrentShape, 1.0));
            var mtBranches = mt.BranchStack(2);
            var mt2 = mt.PradOp.Square();
            var mt2Branch = mt2.Branch();
            var mt3 = mt2.Then(PradOp.MulOp, mtBranches.Pop().BranchInitialTensor);

            var a0 = mt3.PradOp.Mul(new Tensor(t.CurrentShape, 4.0)).PradOp.Mul(tBranches.Pop().BranchInitialTensor);
            var a1 = mt2Branch.Mul(new Tensor(t.CurrentShape, 6.0)).PradOp.Mul(t2Branch.BranchInitialTensor);
            var a2i = mtBranches.Pop().Mul(new Tensor(t.CurrentShape, 4.0));
            var a2 = t3.PradOp.Mul(a2i.Result);

            var r0 = a0 * p0;
            var r1 = a1 * p1;
            var r2 = a2 * p2;

            var cubicBezierResult = r2
                      .Then(PradOp.AddOp, r1.Result)
                      .Then(PradOp.AddOp, r0.Result);
            return cubicBezierResult;
        }

        public PradResult Atan2Approximation(PradOp y, PradOp x,
                                     PradOp alpha, PradOp beta, PradOp lambda, PradOp gamma,
                                     PradOp N_cos, PradOp N_sin,
                                     PradOp p0, PradOp p1, PradOp p2)
        {
            var b1_p0 = p0.Branch();
            var b1_p1 = p1.Branch();
            var b1_p2 = p2.Branch();

            var b2_p0 = p0.Branch();
            var b2_p1 = p1.Branch();
            var b2_p2 = p2.Branch();

            var b3_p0 = p0.Branch();
            var b3_p1 = p1.Branch();
            var b3_p2 = p2.Branch();

            var branchNcos = N_cos.Branch();
            var branchNsin = N_sin.Branch();

            var branchX = x.Branch();
            var branchY = y.Branch();

            var BWCosX = BezierWaveform(x, N_cos, p0, p1, p2);
            var BWCosY = BezierWaveform(y, branchNcos, b1_p0, b1_p1, b1_p2);
            var BWSinX = BezierWaveform(branchX, N_sin, b2_p0, b2_p1, b2_p2);
            var BWSinY = BezierWaveform(branchY, branchNsin, b3_p0, b3_p1, b3_p2);

            var term1 = alpha.Mul(BWCosX.Result);
            var term2 = beta.Mul(BWSinX.Result);
            var term3 = lambda.Mul(BWCosY.Result);
            var term4 = gamma.Mul(BWSinY.Result);

            var result = term1.Then(PradOp.AddOp, term2.Result)
                        .Then(PradOp.AddOp, term3.Result)
                        .Then(PradOp.AddOp, term4.Result);
            var normalized = result.Then(PradOp.ModulusOp, new Tensor(result.PradOp.CurrentShape, 2.0 * Math.PI));
            return normalized;
        }

        [Fact]
        public void TestEmbedding()
        {
            Random rand = new Random(3);
            var input1 = new Tensor(new int[] { 6, 1 }, Enumerable.Range(0, 3).Select(i => (double)i).Concat(Enumerable.Range(0, 3).Select(i => (double)i)).ToArray());
            var embeddings = new Tensor(new int[] { 6, 6 }, Enumerable.Range(0, 36).Select(i => rand.NextDouble()).ToArray());
            var gradientLoss = new Tensor(new int[] { 6, 6 }, Enumerable.Range(0, 36).Select(i => rand.NextDouble() * -1d).ToArray());

            PradOp input1Prad = new PradOp(input1);
            var embeddingRes = input1Prad.Embedding(embeddings);
            embeddingRes.Back(gradientLoss);
        }

        [Fact]
        public void TestEmbedding2()
        {
            Random rand = new Random(3);
            var input1 = new Tensor(new int[] { 2, 6, 1 }, Enumerable.Range(0, 6).Select(i => (double)i).Concat(Enumerable.Range(0, 6).Select(i => (double)i)).ToArray());
            var embeddings = new Tensor(new int[] { 2, 6, 6 }, Enumerable.Range(0, 72).Select(i => rand.NextDouble()).ToArray());
            var gradientLoss = new Tensor(new int[] { 2, 6, 1, 6 }, Enumerable.Range(0, 72).Select(i => rand.NextDouble() * -1d).ToArray());

            PradOp input1Prad = new PradOp(input1);
            var embeddingRes = input1Prad.Embedding(embeddings);
            embeddingRes.Back(gradientLoss);
        }

        [Fact]
        public void TestVNNElementwiseAddBezierOperation()
        {
            Random rand = new Random(3);
            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)i).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            // Create Bezier Waveforms
            var N_sin = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.5, 9).ToArray());
            var N_cos = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.5, 9).ToArray());
            var p0 = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(0.5, 9).ToArray());
            var p1 = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.0, 9).ToArray());
            var p2 = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.5, 9).ToArray());

            // Parameters for Atan2 approximation
            var alpha = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.0, 9).ToArray());
            var beta = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.0, 9).ToArray());
            var lambda = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.0, 9).ToArray());
            var gamma = new Tensor(new int[] { 3, 3 }, Enumerable.Repeat(1.0, 9).ToArray());

            // Naive implementation
            ElementwiseVectorWeightedAddBezierOperation op = new ElementwiseVectorWeightedAddBezierOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix(),
                                          N_sin.ToMatrix(), N_cos.ToMatrix(),
                                          p0.ToMatrix(), p1.ToMatrix(), p2.ToMatrix(),
                                          alpha.ToMatrix(), beta.ToMatrix(), lambda.ToMatrix(), gamma.ToMatrix()).ToTensor();

            // PradOp implementation
            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            // Create branches for magnitudes and angles
            var magnitude1Branch = magnitude1.Branch();
            var magnitude2Branch = magnitude2.Branch();
            var angle1Branch = angle1.Branch();
            var angle2Branch = angle2.Branch();

            // Create PradOp instances for Bezier control points
            var opN_sin = new PradOp(N_sin);
            var opN_cos = new PradOp(N_cos);
            var opP0 = new PradOp(p0);
            var opP1 = new PradOp(p1);
            var opP2 = new PradOp(p2);

            var branchedOpNCos = opN_cos.Branch();
            var branchedOpNSin = opN_sin.Branch();
            var bOpNCos = opN_cos.Branch();
            var bOpNSin = opN_sin.Branch();
            var branchedP0 = opP0.Branch();
            var branchedP1 = opP1.Branch();
            var branchedP2 = opP2.Branch();

            var bP0 = opP0.Branch();
            var bP1 = opP1.Branch();
            var bP2 = opP2.Branch();

            var b4P0 = opP0.Branch();
            var b4P1 = opP1.Branch();
            var b4P2 = opP2.Branch();

            var b5P0 = opP0.Branch();
            var b5P1 = opP1.Branch();
            var b5P2 = opP2.Branch();

            // Compute vector components using Bezier waveforms
            var wave1 = BezierWaveform(angle1, opN_cos, opP0, opP1, opP2).Result;
            var x1 = magnitude1.Mul(wave1);
            var wave2 = BezierWaveform(angle1Branch, opN_sin, branchedP0, branchedP1, branchedP2).Result;
            var y1 = magnitude1Branch.Mul(wave2);
            var wave3 = BezierWaveform(angle2, branchedOpNCos, bP0, bP1, bP2).Result;
            var x2 = magnitude2.Mul(wave3);
            var wave4 = BezierWaveform(angle2Branch, branchedOpNSin, b4P0, b4P1, b4P2).Result;
            var y2 = magnitude2Branch.Mul(wave4);

            // Sum components
            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            // Compute resultant vector magnitude and angle
            var sumXSquared = sumX.PradOp.Square();
            var sumYSquared = sumY.PradOp.Square();
            var magnitudeSquared = sumXSquared.Then(PradOp.AddOp, sumYSquared.Result);
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp)
                                                  .Then(PradOp.MulOp, opWeights.SeedResult.Result);

            // Approximate Atan2 using Bezier waveforms
            var opAlpha = new PradOp(alpha);
            var opBeta = new PradOp(beta);
            var opLambda = new PradOp(lambda);
            var opGamma = new PradOp(gamma);

            var resultAngle = Atan2Approximation(sumYBranch, sumXBranch,
                                                 opAlpha, opBeta, opLambda, opGamma,
                                                 bOpNCos, bOpNSin,
                                                 b5P0, b5P1, b5P2);

            // Concatenate results
            var res = resultMagnitude.PradOp.Concat(new[] { resultAngle.Result }, axis: 1);

            var pradOpOutputCode = res.Result.PrintCode();
            var naiveOutputCode = resultTensor.PrintCode();

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            Assert.Equal(naiveOutputCode, pradOpOutputCode);
        }

        [Fact]
        public void TestVNNElementwiseAddOperationBack()
        {
            GradientRecorder.Instance.RecordingEnabled = true;

            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)i).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            // Compute vector components

            var angle1Branch = angle1.Branch();

            var cosResult = angle1.Cos().Result;
            var sinResult = angle1Branch.Sin().Result;

            var (x1, y1) = magnitude1.DoParallel(
                mag => mag.Mul(cosResult),
                mag => mag.Mul(sinResult)
            );

            var angle2Branch = angle2.Branch();

            var cosResult1 = angle2.Cos().Result;
            var sinResult1 = angle2Branch.Sin().Result;

            var (x2, y2) = magnitude2.DoParallel(
                mag => mag.Mul(cosResult1),
                mag => mag.Mul(sinResult1)
            );

            // Sum components
            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            // Compute resultant vector magnitude and angle
            var sumXSquared = sumX.PradOp.Square();
            var sumYSquared = sumY.PradOp.Square();
            var magnitudeSquared = sumXSquared.Then(PradOp.AddOp, sumYSquared.Result);
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp)
                                                  .Then(PradOp.MulOp, opWeights.SeedResult.Result);
            var resultAngle = sumYBranch.Atan2(sumXBranch.SeedResult.Result);

            // Concatenate results
            var res = resultMagnitude.PradOp.Concat(new[] { resultAngle.Result }, axis: 1);
                
            var gradient = new Tensor(res.Result.Shape, 1.0);
            res.PradOp.Back(gradient);

            var recorded = GradientRecorder.Instance.GetRecordedGradients();
        }

        [Fact]
        public void TestAtan2()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i + 1)).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)((i + 1) * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            var upstream = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)1).ToArray());

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var cos = opInput1.Cos();
            var sin = opInput2.Sin();

            var ss = cos.PradOp.Mul(sin.Result);

            var ssBranch = ss.Branch();

            var sq = ss.PradOp.Exp();

            var sumXBranch = sq.Branch();

            // Compute resultant vector magnitude and angle
            var sumXSquared = sq.PradOp.Square();
            var sumYSquared = opInput2.Square();
            var magnitudeSquared = sumXSquared.Then(PradOp.AddOp, sumYSquared.Result);
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp)
                                                  .Then(PradOp.MulOp, opWeights.SeedResult.Result);
            var resultAngle = ssBranch.Atan2(sumXBranch.SeedResult.Result);

            // Concatenate results
            var res = resultMagnitude.PradOp.Add(resultAngle.Result);

            res.Back(upstream);

        }

        [Fact]
        public void TestVNNElementwiseAddOperation()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i + 1)).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)((i + 1) * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            var upstream = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => ((i + 1) % 10) + rand.NextDouble()).ToArray());

            ElementwiseVectorWeightedAddOperation op = new ElementwiseVectorWeightedAddOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var res1 = op.Backward(upstream.ToMatrix());
            var res21 = res1.Item1 as Matrix;
            var res22 = res1.Item2 as Matrix;
            var res23 = res1.Item3 as Matrix;
            var resTensor1 = res21.ToTensor();
            var resTensor2 = res22.ToTensor();
            var resTensor3 = res23.ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            // Compute vector components

            var magnitude1Branch = magnitude1.Branch();
            var angle1Branch = angle1.Branch();

            var cosResult = angle1.Cos().Result;
            var sinResult = angle1Branch.Sin().Result;

            var x1 = magnitude1.Mul(cosResult);
            var y1 = magnitude1Branch.Mul(sinResult);

            //var (x1, y1) = magnitude1.DoParallel(
            //    mag => mag.Mul(cosResult),
            //    mag => mag.Mul(sinResult)
            //);

            var magnitude2Branch = magnitude2.Branch();
            var angle2Branch = angle2.Branch();

            var cosResult1 = angle2.Cos().Result;
            var sinResult1 = angle2Branch.Sin().Result;

            var x2 = magnitude2.Mul(cosResult1);
            var y2 = magnitude2Branch.Mul(sinResult1);

            //var (x2, y2) = magnitude2.DoParallel(
            //    mag => mag.Mul(cosResult1),
            //    mag => mag.Mul(sinResult1)
            //);

            // Sum components
            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            // Compute resultant vector magnitude and angle
            var sumXSquared = sumX.PradOp.Square();
            var sumYSquared = sumY.PradOp.Square();
            var magnitudeSquared = sumXSquared.Then(PradOp.AddOp, sumYSquared.Result);
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp)
                                                  .Then(PradOp.MulOp, opWeights.SeedResult.Result);
            var resultAngle = sumYBranch.Atan2(sumXBranch.SeedResult.Result);

            // Concatenate results
            var res = resultMagnitude.PradOp.Concat(new[] { resultAngle.Result }, axis: 1);

            var gradientRes = res.Back(upstream);
            var opi1 = opInput1.SeedGradient;
            var opi2 = opInput2.SeedGradient;
            var w1 = opWeights.SeedGradient;

            var pradOpOutputCode = res.Result.PrintCode();
            var naiveOutputCode = resultTensor.PrintCode();

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            Assert.Equal(naiveOutputCode, pradOpOutputCode);
        }

        [Fact]
        public void TestVNNCartesianSummationOperation()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)i).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            var upstream = new Tensor(new int[] { 1, 2 }, Enumerable.Range(0, 2).Select(i => ((i + 1) % 10) + rand.NextDouble()).ToArray());

            ElementwiseVectorCartesianSummationOperation op = new ElementwiseVectorCartesianSummationOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            // Create branches for magnitude1 and magnitude2
            var magnitude1Branch = magnitude1.Branch();
            var magnitude2Branch = magnitude2.Branch();

            var angle1Branch = angle1.Branch();
            var angle2Branch = angle2.Branch();

            // Compute vector components
            var x1 = magnitude1.Mul(angle1.Cos().Result);
            var y1 = magnitude1Branch.Mul(angle1Branch.Sin().Result);
            var x2 = magnitude2.Mul(angle2.Cos().Result);
            var y2 = magnitude2Branch.Mul(angle2Branch.Sin().Result);

            // Sum components
            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            // Compute resultant vector magnitude and angle
            var resultMagnitude = sumX.PradOp.Square()
                .Then(PradOp.AddOp, sumY.PradOp.Square().Result)
                .Then(PradOp.SquareRootOp)
                .Then(PradOp.MulOp, opWeights.SeedResult.Result);
            var resultAngle = sumYBranch.Atan2(sumXBranch.SeedResult.Result);

            // Create branch for resultMagnitude
            var resultMagnitudeBranch = resultMagnitude.PradOp.Branch();
            var resultAngleBranch = resultAngle.PradOp.Branch();

            // Compute final x and y components
            var finalX = resultMagnitude.Then(PradOp.MulOp, resultAngle.PradOp.Cos().Result);
            var finalY = resultMagnitudeBranch.Mul(resultAngleBranch.Sin().Result);

            // Sum across all vectors
            var sumXTotal = finalX.PradOp.Sum(new[] { 0, 1 }).PradOp.Reshape(1, 1);
            var sumYTotal = finalY.PradOp.Sum(new[] { 0, 1 }).PradOp.Reshape(1, 1);

            var cc = sumXTotal.PradOp.Concat(new[] { sumYTotal.Result }, axis: 1);
            var res = cc.Result;

            var pradOpOutputCode = res.PrintCode();
            var naiveOutputCode = resultTensor.PrintCode();

            cc.Back(upstream);
            var g1 = opInput1.SeedGradient;
            var g2 = opInput2.SeedGradient;
            var w = opWeights.SeedGradient;

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            Assert.Equal(naiveOutputCode, pradOpOutputCode);
        }

        [Fact]
        public void TestAddAndAtan2()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i + 1)).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i + 1)).ToArray());
            var upstream = new Tensor(new int[] { 3, 12 }, Enumerable.Range(0, 36).Select(i => (double)(i + 1)).ToArray());

            var opInput1 = new PradOp(input1);
            var opInput1Branch = opInput1.Branch();

            var opInput2 = new PradOp(input2);
            var sqq = opInput2.Square();

            var sq1 = opInput1.Square();
            var sq2 = opInput1Branch.Exp();

            var sum = sq1.PradOp.Add(sq2.Result);

            var atan2 = sum.PradOp.Atan2(sq2.Result);

            var concat = atan2.PradOp.Concat(new Tensor[] { sqq.Result }, 1);

            concat.Back(upstream);
        }

        [Fact]
        public void TestVNNOperationConstituentMultiply()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i + 1)).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)((i + 1) * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            var upstream = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i + 1)).ToArray());

            ElementwiseVectorConstituentMultiplyOperation op = new ElementwiseVectorConstituentMultiplyOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var clonedOpInput1 = opInput1.Branch();
            var clonedOpInput2 = opInput2.Branch();

            //rows = tf.shape(input1)[1]
            //cols = tf.shape(input1)[2]
            //half_cols = cols // 2

            var rows = input1.Shape[0];
            var cols = input1.Shape[1];
            var halfCols = cols / 2;

            //# Slice operations
            //angles_seed = input1[:, :, half_cols:]
            //angles_other = input2[:, :, half_cols:]

            var anglesSeed = opInput1.Indexer(":", $"{halfCols}:").Result;
            var anglesOther = opInput2.Indexer(":", $"{halfCols}:").Result;

            //# Concatenation
            //concat_angles = tf.concat([angles_seed, angles_other], axis = 2)

            var concatAngles = opInput1.Concat(new[] { anglesOther }, axis: 1).Result;

            //# Reshape
            //flat_angles = tf.reshape(concat_angles, [batch_size, 1, -1])

            var flatAngles = opInput1.Reshape(new int[] { 1, -1 }).Result;
            var flatAnglesOp = opInput1.Branch();

            //# Trigonometric operations
            //sin_angles = tf.sin(flat_angles)
            //cos_angles = tf.cos(flat_angles)

            var sinAngles = opInput1.Sin();
            var cosAngles = flatAnglesOp.Cos().Result;

            //# More slicing
            //magnitudes_seed = input1[:, :, :half_cols]
            //magnitudes_other = input2[:, :, :half_cols]

            var magnitudesSeed = clonedOpInput1.Indexer(":", $":{halfCols}").Result;
            var magnitudesOther = clonedOpInput2.Indexer(":", $":{halfCols}").Result;

            //# Another concatenation
            //concat_magnitudes = tf.concat([magnitudes_seed, magnitudes_other], axis = 2)

            var concatMagnitudes = clonedOpInput1.Concat(new[] { magnitudesOther }, axis: 1).Result;

            //# Another reshape
            //flat_magnitudes = tf.reshape(concat_magnitudes, [batch_size, 1, -1])

            var flatMagnitudes = clonedOpInput1.Reshape(new int[] { 1, -1 }).Result;
            var flatMagnitudesOp = clonedOpInput1.Branch();

            //# Multiplication operations
            //Ys = flat_magnitudes * sin_angles
            //Xs = flat_magnitudes * cos_angles

            var ys = sinAngles.PradOp.Mul(flatMagnitudes);
            var xs = flatMagnitudesOp.Mul(cosAngles).Result;

            //reshaped_Ys = tf.reshape(Ys, [batch_size, rows, cols])
            //reshaped_Xs = tf.reshape(Xs, [batch_size, rows, cols])

            var reshapedYs = ys.PradOp.Reshape(new int[] { rows, cols });
            var reshapedXs = flatMagnitudesOp.Reshape(new int[] { rows, cols }).Result;
            var reshapedYsOp = ys.PradOp.Branch();
            var reshapedXsOp = flatMagnitudesOp.Branch();

            //y1s = reshaped_Ys[:, :, :half_cols]
            //y2s = reshaped_Ys[:, :, half_cols:]
            //x1s = reshaped_Xs[:, :, :half_cols]
            //x2s = reshaped_Xs[:, :, half_cols:]

            var y1s = reshapedYs.PradOp.Indexer(":", $":{halfCols}");
            var y2s = reshapedYsOp.Indexer(":", $"{halfCols}:").Result;
            var x1s = flatMagnitudesOp.Indexer(":", $":{halfCols}").Result;
            var x2s = reshapedXsOp.Indexer(":", $"{halfCols}:").Result;


            // For X2 and Y2
            var x2Reshaped = reshapedXsOp.Transpose(new int[] { 1, 0 }).Result;
            var x2Tiled = reshapedXsOp.Tile(new int[] { 3, 1 }).Result;
            var flatX2s = reshapedXsOp.Reshape(new int[] { 1, 27 }).Result;

            var y2Reshaped = reshapedYsOp.Transpose(new int[] { 1, 0 }).Result;
            var y2Tiled = reshapedYsOp.Tile(new int[] { 3, 1 });
            var flatY2s = reshapedYsOp.Reshape(new int[] { 1, 27 });

            // For X1 and Y1
            var x1Tiled = flatMagnitudesOp.Tile(new int[] { 1, 3 }).Result;
            var flatX1s = flatMagnitudesOp.Reshape(new int[] { 1, 27 });

            var y1Tiled = reshapedYs.PradOp.Tile(new int[] { 1, 3 }).Result;
            var flatY1s = reshapedYs.PradOp.Reshape(new int[] { 1, 27 });


            //delta_Y = flat_Y2s - flat_Y1s
            //delta_X = flat_X2s - flat_X1s

            Debug.WriteLine("flatX2s: " + reshapedXsOp.PrintCodeForCurrentTensor());
            Debug.WriteLine("flatX1s: " + flatX1s.Result.PrintCode());

            var deltaY = flatY1s.PradOp.SubFrom(reshapedYsOp.Result);
            var deltaX = flatX1s.PradOp.SubFrom(reshapedXsOp.Result);
            var deltaYOp = deltaY.Branch();

            Debug.WriteLine("deltaY: " + deltaY.Result.PrintCode());
            Debug.WriteLine("deltaX: " + deltaX.Result.PrintCode());

            //squared_Y = delta_Y * delta_Y

            //squared_X = delta_X * delta_X

            var squaredY = deltaY.PradOp.Square();
            var squaredX = deltaX.PradOp.Square().Result;

            //added_YX = squared_Y + squared_X

            var addedYX = squaredY.PradOp.Add(squaredX);

            //unweighted_magnitude = tf.sqrt(added_YX)

            var unweightedMagnitude = addedYX.PradOp.SquareRoot();

            //transposed_weights = tf.transpose(weights, [0, 2, 1])
            //tiled_weights = tf.tile(transposed_weights, [1, 3, 1])
            //flattened_weights = tf.reshape(tiled_weights, [batch_size, 1, -1])

            var transposedWeights = opWeights.Transpose(new int[] { 1, 0 }).Result;
            var tiledWeights = opWeights.Tile(new int[] { 3, 1 }).Result;
            var flattenedWeights = opWeights.Reshape(new int[] { 1, -1 }).Result;

            Debug.WriteLine("unweighted mag: " + unweightedMagnitude.Result.PrintCode());
            Debug.WriteLine("flattened weights: " + flattenedWeights.PrintCode());

            //magnitudes = flattened_weights * unweighted_magnitude

            var magnitudes = unweightedMagnitude.PradOp.Mul(flattenedWeights);
            var magnitudesOp = magnitudes.Branch();

            Debug.WriteLine("magnitudes: " + magnitudes.Result.PrintCode());

            //angles = tf.atan2(delta_Y, delta_X)

            var angles = deltaYOp.Atan2(deltaX.Result).Result;
            var anglesOp = deltaYOp.Branch();

            Debug.WriteLine("angles: " + angles.PrintCode());

            //sin_angles2 = tf.sin(angles)
            //cos_angles2 = tf.cos(angles)

            var sinAngles2 = deltaYOp.Sin().Result;
            var cosAngles2 = anglesOp.Cos().Result;

            //y_overall = magnitudes * sin_angles2
            //x_overall = magnitudes * cos_angles2

            var yOverall = magnitudes.PradOp.Mul(sinAngles2);
            var xOverall = magnitudesOp.Mul(cosAngles2);

            //reshaped_Y_overall = tf.reshape(y_overall, [batch_size, rows * half_cols, 3])
            //reshaped_X_overall = tf.reshape(x_overall, [batch_size, rows * half_cols, 3])

            var reshapedYOverall = yOverall.PradOp.Reshape(new int[] { rows * halfCols, 3 });
            var reshapedXOverall = xOverall.PradOp.Reshape(new int[] { rows * halfCols, 3 });

            Debug.WriteLine("reshapedY: " + reshapedYOverall.Result.PrintCode());

            //sum_rows_Y = tf.reduce_sum(reshaped_Y_overall, axis = 2)
            //sum_rows_X = tf.reduce_sum(reshaped_X_overall, axis = 2)

            var sumRowsY = reshapedYOverall.PradOp.SumRows();
            var sumRowsX = reshapedXOverall.PradOp.SumRows();

            Debug.WriteLine("sumRowsY: " + sumRowsY.Result.PrintCode());
            Debug.WriteLine("sumRowsX: " + sumRowsX.Result.PrintCode());

            //flattened_sum_rows_Y = tf.reshape(sum_rows_Y, [batch_size, 1, -1])
            //flattened_sum_rows_X = tf.reshape(sum_rows_X, [batch_size, 1, -1])

            var flattenedSumRowsY = sumRowsY.PradOp.Reshape(new int[] { 1, -1 });
            var flattenedSumRowsX = sumRowsX.PradOp.Reshape(new int[] { 1, -1 });
            var flattenedSumRowsYOp = flattenedSumRowsY.Branch();
            var flattenedSumRowsXOp = flattenedSumRowsX.Branch();

            //flattened_Y_squared = flattened_sum_rows_Y * flattened_sum_rows_Y
            //flattened_X_squared = flattened_sum_rows_X * flattened_sum_rows_X

            var flattenedYSquared = flattenedSumRowsY.PradOp.Square();
            var flattenedXSquared = flattenedSumRowsX.PradOp.Square();

            //added_YX_overall = flattened_Y_squared + flattened_X_squared
            var addedYXOverall = flattenedYSquared.PradOp.Add(flattenedXSquared.Result);

            //magnitudes_overall = tf.sqrt(added_YX_overall)
            //angles_overall = tf.atan2(flattened_sum_rows_Y, flattened_sum_rows_X)

            var magnitudesOverall = addedYXOverall.PradOp.SquareRoot();
            var anglesOverall = flattenedSumRowsYOp.Atan2(flattenedSumRowsXOp.BranchInitialTensor);

            // reshaped_magnitudes_overall = tf.reshape(magnitudes_overall, [batch_size, rows, half_cols])
            // reshaped_angles_overall = tf.reshape(angles_overall, [batch_size, rows, half_cols])

            var reshapedMagnitudesOverall = magnitudesOverall.PradOp.Reshape(new int[] { rows, halfCols });
            var reshapedAnglesOverall = anglesOverall.PradOp.Reshape(new int[] { rows, halfCols }).Result;

            //output_tensor = tf.concat([magnitudes_overall, angles_overall], axis = 2)
            //output_tensor = tf.reshape(output_tensor, [batch_size, rows, cols])

            var outputTensor = reshapedMagnitudesOverall.PradOp.Concat(new[] { reshapedAnglesOverall }, axis: 1);
            var o = outputTensor.PradOp.Reshape(new int[] { rows, cols });
            var output = o.Result;

            var naiveOutput = resultTensor;

            var pradOpOutputCode = output.PrintCode();
            var naiveOutputCode = naiveOutput.PrintCode();

            o.Back(upstream);
            var o1 = opInput1.SeedGradient;
            var o2 = opInput2.SeedGradient;
            var ow = opWeights.SeedGradient;

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            Assert.Equal(naiveOutputCode, pradOpOutputCode);

        }

        [Fact]
        public void TestVNNOperation()
        {
            Random rand = new Random(3);

            // Create large tensors (200x400)
            var seed = new Tensor(new int[] { 200, 400 }, Enumerable.Range(0, 80000).Select(i => (double)i).ToArray());
            var other = new Tensor(new int[] { 200, 400 }, Enumerable.Range(0, 80000).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 200, 200 }, Enumerable.Range(0, 40000).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            var opSeed = new PradOp(seed);
            var opOther = new PradOp(other);

            var clonedOpSeed = opSeed.DeepClone();
            var clonedOther = opOther.DeepClone();

            // Slice operations
            var anglesSeed = opSeed.Slice(new int[] { 0, 200 }, new int[] { 200, 200 }).Result;
            var anglesOther = opOther.Slice(new int[] { 0, 200 }, new int[] { 200, 200 }).Result;

            // Concatenation
            var concatAngles = opSeed.Concat(new[] { anglesOther }, axis: 1).Result;

            // Reshape
            var flatAngles = opSeed.Reshape(new int[] { 1, 80000 }).Result;

            // Trigonometric operations
            var clonedFlatAnglesOp = opSeed.DeepClone();
            var sinAngles = opSeed.Sin().Result;
            var cosAngles = clonedFlatAnglesOp.Cos().Result;

            // More slicing
            var magnitudesSeed = clonedOpSeed.Slice(new int[] { 0, 0 }, new int[] { 200, 200 }).Result;
            var magnitudesOther = clonedOther.Slice(new int[] { 0, 0 }, new int[] { 200, 200 }).Result;

            // Another concatenation
            var concatMagnitudes = clonedOpSeed.Concat(new[] { magnitudesOther }, axis: 1).Result;

            // Another reshape
            var flatMagnitudes = clonedOpSeed.Reshape(new int[] { 1, 80000 }).Result;

            // Multiplication operations
            var clonedFlatMagnitudesOp = clonedOpSeed.DeepClone();
            var Ys = clonedOpSeed.Mul(sinAngles).Result;
            var Xs = clonedFlatMagnitudesOp.Mul(cosAngles).Result;

            var reshapedYs = clonedOpSeed.Reshape(new int[] { 200, 400 }).Result;

            var reshapedOp = clonedOpSeed.DeepClone();

            var y1s = clonedOpSeed.Slice(new int[] { 0, 0 }, new int[] { 200, 200 }).Result;

            var y2s = reshapedOp.Slice(new int[] { 0, 200 }, new int[] { 200, 200 }).Result;

            var reshapedXs = clonedFlatMagnitudesOp.Reshape(new int[] { 200, 400 }).Result;
            
            var reshapedOp2 = clonedFlatMagnitudesOp.DeepClone();

            var x1s = clonedFlatMagnitudesOp.Slice(new int[] { 0, 0 }, new int[] { 200, 200 }).Result;

            var x2s = reshapedOp2.Slice(new int[] { 0, 200 }, new int[] { 200, 200 }).Result;

            var transposedY2s = reshapedOp.Transpose(1, 0).Result;

            var transposedX2s = reshapedOp2.Transpose(1, 0).Result;

            var tiledY2s = reshapedOp.Tile(new int[] { 1, 3 }).Result;

            var tiledX2s = reshapedOp2.Tile(new int[] { 1, 3 }).Result;

            var flatY2s = reshapedOp.Reshape(new int[] { 1, 120000 }).Result;

            var flatX2s = reshapedOp2.Reshape(new int[] { 1, 120000 }).Result;

            var tiledY1s = clonedOpSeed.Tile(new int[] { 1, 3 }).Result;

            var tiledX1s = clonedFlatMagnitudesOp.Tile(new int[] { 1, 3 }).Result;

            var flatY1s = clonedOpSeed.Reshape(new int[] { 1, 120000 }).Result;

            var flatX1s = clonedFlatMagnitudesOp.Reshape(new int[] { 1, 120000 }).Result;

            var deltaY = reshapedOp.Sub(flatY1s).Result;

            var deltaYOp = reshapedOp.DeepClone();

            var deltaX = reshapedOp2.Sub(flatX1s).Result;

            var squaredY = reshapedOp.Mul(deltaY).Result;

            var squaredX = reshapedOp2.Mul(deltaX).Result;

            var addedYX = reshapedOp.Add(squaredX).Result;

            var unweightedMagnitude = reshapedOp.SquareRoot().Result;

            var opWeights = new PradOp(weights);

            var transposedWeights = opWeights.Transpose(1, 0).Result;

            var tiledWeights = opWeights.Tile(new int[] { 1, 3 }).Result;

            var flattenedWeights = opWeights.Reshape(new int[] { 1, 120000 }).Result;

            var magnitudes = opWeights.Mul(unweightedMagnitude).Result;

            var magnitudesOp = opWeights.DeepClone();

            var angles = deltaYOp.Atan2(deltaX).Result;

            var anglesOp = deltaYOp.DeepClone();    

            var sinAngles2 = deltaYOp.Sin().Result;

            var cosAngles2 = anglesOp.Cos().Result;

            var yOverall = opWeights.Mul(sinAngles2).Result;

            var xOverall = magnitudesOp.Mul(cosAngles2).Result;

            var reshapedYOverall = opWeights.Reshape(new int[] { 40000, 3 }).Result;

            var reshapedXOverall = magnitudesOp.Reshape(new int[] { 40000, 3 }).Result;

            var sumRowsY = opWeights.SumRows().Result;

            var sumRowsX = magnitudesOp.SumRows().Result;

            var flattenedSumRowsY = opWeights.Reshape(new int[] { 1, 40000 }).Result;

            var flattenedSumRowsYOp = opWeights.DeepClone();

            var flattenedSumRowsX = magnitudesOp.Reshape(new int[] { 1, 40000 }).Result;

            var flattenedYSquared = opWeights.Mul(flattenedSumRowsY).Result;

            var flattenedXSquared = magnitudesOp.Mul(flattenedSumRowsX).Result;

            var addedYXOverall = opWeights.Add(flattenedXSquared).Result;

            var magnitudesOverall = opWeights.SquareRoot().Result;

            var anglesOverall = flattenedSumRowsYOp.Atan2(flattenedSumRowsX).Result;
        }

        [Fact]
        public void TestConcatAlongAxis1()
        {
            // Create initial tensor
            var seed = new Tensor(new int[] { 3, 2 }, new double[] { 1, 2, 3, 4, 5, 6 });
            var pradOp = new PradOp(seed);

            // Create tensor to concatenate
            var tensorToConcat = new Tensor(new int[] { 3, 3 }, new double[] { 7, 8, 9, 10, 11, 12, 13, 14, 15 });

            // Perform concatenation along axis 1
            var result = pradOp.Concat(new Tensor[] { tensorToConcat }, axis: 1);

            // Assert the shape of the result
            Assert.Equal(new int[] { 3, 5 }, result.Result.Shape);

            // Assert the values of the concatenated tensor
            Assert.Equal(new double[] {
                1, 2, 7, 8, 9,
                3, 4, 10, 11, 12,
                5, 6, 13, 14, 15
            }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 3, 5 }, new double[] {
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1
            });
            pradOp.Back(upstreamGradient);

            // Assert the gradients for the original tensor
            Assert.Equal(new double[] { 1, 1, 1, 1, 1, 1 }, result.Gradients[0].Data);

            // Assert the gradients for the concatenated tensor
            Assert.Equal(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 }, result.Gradients[1].Data);

            // Print the result for debugging
            Console.WriteLine($"Result shape: [{string.Join(", ", result.Result.Shape)}]");
            Console.WriteLine($"Result data: [{string.Join(", ", result.Result.Data)}]");
            Console.WriteLine($"Gradient 0 data: [{string.Join(", ", result.Gradients[0].Data)}]");
            Console.WriteLine($"Gradient 1 data: [{string.Join(", ", result.Gradients[1].Data)}]");
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

            var result = pradOp.CustomOperation(operation, reverseOperation);

            Assert.Equal(new double[] { 1, 4, 9, 16 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 2, 4, 6, 8 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestCustomOperation2()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
            PradOp pradOp = new PradOp(seed);

            // Define a custom operation (e.g., element-wise square and add)
            TensorOp customOperation = tensor =>
            {
                PradOp pradOp = new PradOp(tensor[0]);
                pradOp
                    .Square()
                    .PradOp.Add(tensor[1]);
                return pradOp;
            };

            var result = pradOp.CustomOperation(customOperation, new Tensor[] { new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 }) });

            Assert.Equal(new double[] { 2, 6, 12, 20 }, pradOp.Result!.Data);

            result.Then(PradOp.CustomTensorOp, customOperation, new Tensor[] { new Tensor(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 }) });

            Assert.Equal(new double[] { 5, 38, 147, 404 }, pradOp.Result!.Data);
        }

        [Fact]
        public void TestThenCustomOp()
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

            var result = pradOp.Square()
                        .Then(PradOp.CustomOp, operation, reverseOperation);

            Assert.Equal(new double[] { 1, 16, 81, 256 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 2, 8, 18, 32 }, result.Gradients[0].Data);
        }

        [Fact]
        public void TestElementwiseSubFrom()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 5, 6, 7, 8 });
            var tensorToSubFrom = new Tensor(new int[] { 2, 2 }, new double[] { 10, 11, 12, 13 });
            var pradOp = new PradOp(seed);

            var result = pradOp.SubFrom(tensorToSubFrom);

            Assert.Equal(new double[] { 5, 5, 5, 5 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { -1, -1, -1, -1 }, result.Gradients[0].Data);
            Assert.Equal(new double[] { 1, 1, 1, 1 }, result.Gradients[1].Data);
        }

        [Fact]
        public void TestElementwiseDivInto()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 2, 4, 8, 16 });
            var tensorToDivInto = new Tensor(new int[] { 2, 2 }, new double[] { 10, 20, 40, 80 });
            var pradOp = new PradOp(seed);

            var result = pradOp.DivInto(tensorToDivInto);

            Assert.Equal(new double[] { 5, 5, 5, 5 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            // The gradient for the denominator (seed) should be -numerator / denominator^2
            Assert.Equal(new double[] { -2.5, -1.25, -0.625, -0.3125 }, result.Gradients[0].Data);

            // The gradient for the numerator (tensorToDivInto) should be 1 / denominator
            Assert.Equal(new double[] { 0.5, 0.25, 0.125, 0.0625 }, result.Gradients[1].Data);
        }

        [Fact]
        public void TestThenParallel()
        {
            var seed = new Tensor(new int[] { 2, 2 }, new double[] { 0.2d, 0.4d, 0.8d, 0.16d });
            var otherTensor = new Tensor(new int[] { 2, 2 }, new double[] { 0.1d, 0.2d, 0.4d, 0.8d });
            var pradOp = new PradOp(seed);

            var result1 = pradOp.Square()
                .ThenParallel(result => result.PradOp.Add(otherTensor),
                    result => result.PradOp.Mul(otherTensor),
                    result => result.PradOp.SubFrom(otherTensor))
                .Then(resultArray => resultArray[0].PradOp.Add(resultArray[1].Result).PradOp.SubFrom(resultArray[2].Result));

            var pradOp2 = new PradOp(seed);

            var result2 = pradOp2.Square()
                .ThenParallel(result => result.PradOp.Add(otherTensor),
                    result => result.PradOp.Mul(otherTensor),
                    result => result.PradOp.SubFrom(otherTensor))
                .Then(resultArray =>
                {
                    var result1 = resultArray[0].PradOp.Add(resultArray[1].Result);
                    var result2 = resultArray[2].PradOp.SubFrom(resultArray[1].Result);
                    return new[] { result1, result2 };
                })
                .Then(resultArray => { 
                    return resultArray[0].PradOp.Add(resultArray[1].Result);
                });

            var result3 = result1 * result2;

            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });

            var gradient = pradOp.Back(upstreamGradient);
        }

        [Fact]
        public void TestPow()
        {
            var input = new Tensor(new int[] { 1, 4 }, new double[] { 0.1, 0.2, 0.3, 0.4 });
            var pradOp = new PradOp(input);
            var result = pradOp.Pow(2f).Then(PradOp.PowOp, new Tensor(input.Shape, 5d));
        }

        [Fact]
        public void TestThenParallel2()
        {
            // Define input and weights
            var input = new Tensor(new int[] { 1, 4 }, new double[] { 0.1, 0.2, 0.3, 0.4 });
            var weights = new Tensor(new int[] { 4, 3 }, new double[] {
                0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
                0.7, 0.8, 0.9,
                1.0, 1.1, 1.2
            });
            var bias = new Tensor(new int[] { 1, 3 }, new double[] { 0.1, 0.2, 0.3 });

            // Create PradOp instance
            var pradOp = new PradOp(input);

            PradResult? weightsResult = null;
            PradResult? biasResult = null;

            // Compute layer output with multiple activations
            var result = pradOp.MatMul(weights)
                .Then(result => {
                    weightsResult = result;
                    return result.Then(PradOp.AddOp, bias);
                })
                .Then(result => {
                    biasResult = result;
                    return result.ThenParallel(
                        result => result.Then(PradOp.SinOp),       // Sine activation
                        result => result.Then(PradOp.ReciprocalOp).Then(PradOp.AddOp, new Tensor(new int[] { 1, 3 }, 1)),
                        result => result.Then(PradOp.ExpOp));        // Exponential activation
                })
                .Then(activations => {
                    // Compute weighted sum of activations
                    var weights = new Tensor(new int[] { 3 }, new double[] { 0.3, 0.3, 0.4 });
                    return activations
                        .Select((act, i) => act.PradOp.Mul(weights.Indexer($"{i}").BroadcastTo(new int[] { 1, 3 })))
                        .Aggregate((a, b) => a.PradOp.Add(b.Result));
                });

            // Compute gradient
            var upstreamGradient = new Tensor(new int[] { 1, 3 }, new double[] { 1, 1, 1 });
            var gradient = pradOp.Back(upstreamGradient);

            // Access results and gradients
            Console.WriteLine("Layer output: " + result.Result);
            Console.WriteLine("Input gradient: " + gradient);
        }
    }
}
