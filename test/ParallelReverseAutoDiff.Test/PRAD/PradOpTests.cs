using ILGPU.Runtime.Cuda;
using MKLNET;
using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using System.Diagnostics;
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
            var flatAnglesOp = opInput1.Branch();
            flatAnglesOp.SetUpstreamGradient(new Tensor(new int[] { 1, 18 }, Enumerable.Range(0, 18).Select(x => rand.NextDouble()).ToArray()));

            var sinAngles = opInput1.Sin();
            var cosAngles = flatAnglesOp.Cos();

            var dInput1 = opInput1.Back(new Tensor(new int[] { 1, 18 }, Enumerable.Range(0, 18).Select(x => rand.NextDouble()).ToArray()));
        }

        [Fact]
        public void TestVNNDecompositionOperationUsingIndexer()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => i / 100d).ToArray());
            var input2 = new Tensor(new int[] { 3, 20 }, Enumerable.Range(0, 60).Select(i => (i * 2) / 100d).ToArray());
            var weights = new Tensor(new int[] { 3, 2 }, Enumerable.Range(0, 6).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            ElementwiseVectorDecompositionOperation op = new ElementwiseVectorDecompositionOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);
            var opInput1Branch = opInput1.Branch();
            var opInput2Branch = opInput2.Branch();
            var opInput2Branch2 = opInput2.Branch();

            //            num_rows, num_cols = tf.shape(input1)[1], tf.shape(input1)[2] // 2

            var num_rows = input1.Shape[0];
            var num_cols = input1.Shape[1] / 2;

            //        # Split input1 into magnitude and angle
            //        magnitude = input1[:, :, :num_cols]
            //        angle = input1[:, :, num_cols:]

            var magnitudes = opInput1.Indexer(":", $":{num_cols}");
            var angles = opInput1Branch.Indexer(":", $"{num_cols}:");
            var anglesBranch = opInput1Branch.Branch();
            var magnitudesBranch = opInput1.Branch();

            //        # Extract components from input2
            //        input2_cols = tf.shape(input2)[2]
            //        half_cols = input2_cols // 2

            var input2_cols = input2.Shape[1];
            var half_cols = input2_cols / 2;

            //        # Correctly extract w_magnitude_pivot and w_angle_pivot
            //        w_magnitude_pivot = input2[:, :, :half_cols][:, :, ::5]
            //        w_angle_pivot = input2[:, :, half_cols:][:, :, ::5]

            var w_magnitude_pivot_half = opInput2.Indexer(":", $":{half_cols}");
            var w_magnitude_pivot = opInput2.Indexer(":", $"::5");

            var w_angle_pivot_half = opInput2Branch.Indexer(":", $"{half_cols}:");
            var w_angle_pivot = opInput2Branch.Indexer(":", $"::5");

            //        # Extract other components
            //        w_magnitudes = tf.stack([input2[:, :, 1 + i:half_cols: 5] for i in range(4)], axis = -1)
            //                w_angles = tf.stack([input2[:, :, half_cols + 1 + i::5] for i in range(4)], axis = -1)

            var w_magnitudes_t = new Tensor[4];
            var w_angles_t = new Tensor[4];
            for (int i = 0; i < 4; i++)
            {
                var branchM = opInput2Branch2.DeepClone();
                w_magnitudes_t[i] = branchM.Indexer(":", $"{1 + i}:{half_cols}:5").Result;

                var branchA = opInput2Branch2.DeepClone();
                w_angles_t[i] = branchA.Indexer(":", $"{half_cols + 1 + i}::5").Result;
            }
            var w_magnitudes_stacked = new PradOp(w_magnitudes_t[0]).Stack(w_magnitudes_t.Skip(1).ToArray(), axis: -1);
            var w_angles_stacked = new PradOp(w_angles_t[0]).Stack(w_angles_t.Skip(1).ToArray(), axis: -1);

            var w_magnitudes = w_magnitudes_stacked.PradOp;
            var w_angles = w_angles_stacked.PradOp;

            //# Compute x and y components
            //                x = magnitude * tf.math.cos(angle)
            //        y = magnitude * tf.math.sin(angle)

            var cosAngles = opInput1Branch.Cos();
            var sinAngles = anglesBranch.Sin();

            var (x, y) = opInput1.DoMultiple(
                x => x.Mul(cosAngles.Result), 
                x => x.Mul(sinAngles.Result));

            //        x_pivot = w_magnitude_pivot * tf.math.cos(w_angle_pivot)
            //        y_pivot = w_magnitude_pivot * tf.math.sin(w_angle_pivot)

            var (cosPivot, sinPivot) = opInput2Branch.DoMultiple(
                x => x.Cos(),
                x => x.Sin());

            var (xPivot, yPivot) = opInput2.DoMultiple(
                x => x.Mul(cosPivot.Result),
                x => x.Mul(sinPivot.Result));

            //        x_w = w_magnitudes * tf.math.cos(w_angles)
            //        y_w = w_magnitudes * tf.math.sin(w_angles)

            var (cosAngles_w, sinAngles_w) = w_angles.DoMultiple(
                x => x.Cos(),
                x => x.Sin());

            var (x_w, y_w) = w_magnitudes.DoMultiple(
                x => x.Mul(cosAngles_w.Result),
                x => x.Mul(sinAngles_w.Result));

            //        # Adjust weights
            //        weights = tf.where(tf.abs(weights) < 0.01, tf.sign(weights) * 0.01, weights)

            //        # Compute sum components
            //        sum_x = (x + x_pivot) / (weights + 1e-9)
            //        sum_y = (y + y_pivot) / (weights + 1e-9)

            //        # Compute differences
            //        sum_x_expanded = tf.expand_dims(sum_x, -1)
            //        sum_y_expanded = tf.expand_dims(sum_y, -1)


            //        diff_x = tf.concat([
            //            sum_x_expanded - x_w[:, :, :, 0:1],
            //            -sum_x_expanded - x_w[:, :, :, 1:2],
            //            sum_x_expanded - x_w[:, :, :, 2:3],
            //            -sum_x_expanded - x_w[:, :, :, 3:4]
            //        ], axis = -1)


            //        diff_y = tf.concat([
            //            sum_y_expanded - y_w[:, :, :, 0:1],
            //            -sum_y_expanded - y_w[:, :, :, 1:2],
            //            sum_y_expanded - y_w[:, :, :, 2:3],
            //            -sum_y_expanded - y_w[:, :, :, 3:4]
            //        ], axis = -1)

            //        # Compute result magnitudes and angles
            //        result_magnitudes = tf.math.sqrt(diff_x * *2 + diff_y * *2)
            //        result_angles = tf.math.atan2(diff_y, diff_x)

            //        # Combine all results
            //        output = tf.concat([
            //            magnitude, w_magnitude_pivot,
            //            tf.reshape(w_magnitudes, [batch_size, num_rows, -1]),
            //            tf.reshape(result_magnitudes, [batch_size, num_rows, -1]),
            //            angle, w_angle_pivot,
            //            tf.reshape(w_angles, [batch_size, num_rows, -1]),
            //            tf.reshape(result_angles, [batch_size, num_rows, -1])
            //        ], axis = -1)



            /*
            var pradOpOutputCode = output.PrintCode();
            var naiveOutputCode = naiveOutput.PrintCode();

            Debug.WriteLine("pradOpOutput: " + pradOpOutputCode);
            Debug.WriteLine("naiveOutput: " + naiveOutputCode);

            Assert.Equal(naiveOutputCode, pradOpOutputCode);
            */
        }

        [Fact]
        public void TestVNNOperationUsingIndexer()
        {
            Random rand = new Random(3);

            var input1 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)i).ToArray());
            var input2 = new Tensor(new int[] { 3, 6 }, Enumerable.Range(0, 18).Select(i => (double)(i * 2)).ToArray());
            var weights = new Tensor(new int[] { 3, 3 }, Enumerable.Range(0, 9).Select(i => (i % 10) + rand.NextDouble()).ToArray());

            ElementwiseVectorConstituentMultiplyOperation op = new ElementwiseVectorConstituentMultiplyOperation();
            var resultTensor = op.Forward(input1.ToMatrix(), input2.ToMatrix(), weights.ToMatrix()).ToTensor();

            var opInput1 = new PradOp(input1);
            var opInput2 = new PradOp(input2);
            var opWeights = new PradOp(weights);

            var clonedOpInput1 = opInput1.DeepClone();
            var clonedOpInput2 = opInput2.DeepClone();

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
            var flatAnglesOp = opInput1.DeepClone();

            //# Trigonometric operations
            //sin_angles = tf.sin(flat_angles)
            //cos_angles = tf.cos(flat_angles)

            var sinAngles = opInput1.Sin().Result;
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
            var flatMagnitudesOp = clonedOpInput1.DeepClone();

            //# Multiplication operations
            //Ys = flat_magnitudes * sin_angles
            //Xs = flat_magnitudes * cos_angles

            var ys = clonedOpInput1.Mul(sinAngles).Result;
            var xs = flatMagnitudesOp.Mul(cosAngles).Result;

            //reshaped_Ys = tf.reshape(Ys, [batch_size, rows, cols])
            //reshaped_Xs = tf.reshape(Xs, [batch_size, rows, cols])

            var reshapedYs = clonedOpInput1.Reshape(new int[] { rows, cols }).Result;
            var reshapedXs = flatMagnitudesOp.Reshape(new int[] { rows, cols }).Result;
            var reshapedYsOp = clonedOpInput1.DeepClone();
            var reshapedXsOp = flatMagnitudesOp.DeepClone();

            //y1s = reshaped_Ys[:, :, :half_cols]
            //y2s = reshaped_Ys[:, :, half_cols:]
            //x1s = reshaped_Xs[:, :, :half_cols]
            //x2s = reshaped_Xs[:, :, half_cols:]

            var y1s = clonedOpInput1.Indexer(":", $":{halfCols}").Result;
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
            var flatX1s = flatMagnitudesOp.Reshape(new int[] { 1, 27 }).Result;

            var y1Tiled = clonedOpInput1.Tile(new int[] { 1, 3 }).Result;
            var flatY1s = clonedOpInput1.Reshape(new int[] { 1, 27 }).Result;


            //delta_Y = flat_Y2s - flat_Y1s
            //delta_X = flat_X2s - flat_X1s

            Debug.WriteLine("flatX2s: " + reshapedXsOp.PrintCodeForCurrentTensor());
            Debug.WriteLine("flatX1s: " + flatX1s.PrintCode());

            var deltaY = reshapedYsOp.Sub(flatY1s).Result;
            var deltaX = reshapedXsOp.Sub(flatX1s).Result;
            var deltaYOp = reshapedYsOp.DeepClone();

            Debug.WriteLine("deltaY: " + deltaY.PrintCode());
            Debug.WriteLine("deltaX: " + deltaX.PrintCode());

            //squared_Y = delta_Y * delta_Y

            //squared_X = delta_X * delta_X

            var squaredY = reshapedYsOp.Mul(deltaY).Result;
            var squaredX = reshapedXsOp.Mul(deltaX).Result;

            //added_YX = squared_Y + squared_X

            var addedYX = reshapedYsOp.Add(squaredX).Result;

            //unweighted_magnitude = tf.sqrt(added_YX)

            var unweightedMagnitude = reshapedYsOp.SquareRoot().Result;

            //transposed_weights = tf.transpose(weights, [0, 2, 1])
            //tiled_weights = tf.tile(transposed_weights, [1, 3, 1])
            //flattened_weights = tf.reshape(tiled_weights, [batch_size, 1, -1])

            var transposedWeights = opWeights.Transpose(new int[] { 1, 0 }).Result;
            var tiledWeights = opWeights.Tile(new int[] { 3, 1 }).Result;
            var flattenedWeights = opWeights.Reshape(new int[] { 1, -1 }).Result;
            var flattenedWeightsOp = opWeights.DeepClone();

            Debug.WriteLine("unweighted mag: " + unweightedMagnitude.PrintCode());
            Debug.WriteLine("flattened weights: " + flattenedWeights.PrintCode());

            //magnitudes = flattened_weights * unweighted_magnitude

            var magnitudes = opWeights.Mul(unweightedMagnitude).Result;
            var magnitudesOp = opWeights.DeepClone();

            Debug.WriteLine("magnitudes: " + magnitudes.PrintCode());

            //angles = tf.atan2(delta_Y, delta_X)

            var angles = deltaYOp.Atan2(deltaX).Result;
            var anglesOp = deltaYOp.DeepClone();

            Debug.WriteLine("angles: " + angles.PrintCode());

            //sin_angles2 = tf.sin(angles)
            //cos_angles2 = tf.cos(angles)

            var sinAngles2 = deltaYOp.Sin().Result;
            var cosAngles2 = anglesOp.Cos().Result;

            //y_overall = magnitudes * sin_angles2
            //x_overall = magnitudes * cos_angles2

            var yOverall = magnitudesOp.Mul(sinAngles2).Result;
            var xOverall = opWeights.Mul(cosAngles2).Result;

            //reshaped_Y_overall = tf.reshape(y_overall, [batch_size, rows * half_cols, 3])
            //reshaped_X_overall = tf.reshape(x_overall, [batch_size, rows * half_cols, 3])

            var reshapedYOverall = magnitudesOp.Reshape(new int[] { rows * halfCols, 3 }).Result;
            var reshapedXOverall = opWeights.Reshape(new int[] { rows * halfCols, 3 }).Result;

            Debug.WriteLine("reshapedY: " + reshapedYOverall.PrintCode());

            //sum_rows_Y = tf.reduce_sum(reshaped_Y_overall, axis = 2)
            //sum_rows_X = tf.reduce_sum(reshaped_X_overall, axis = 2)

            var sumRowsY = magnitudesOp.SumRows().Result;
            var sumRowsX = opWeights.SumRows().Result;

            Debug.WriteLine("sumRowsY: " + sumRowsY.PrintCode());
            Debug.WriteLine("sumRowsX: " + sumRowsX.PrintCode());

            //flattened_sum_rows_Y = tf.reshape(sum_rows_Y, [batch_size, 1, -1])
            //flattened_sum_rows_X = tf.reshape(sum_rows_X, [batch_size, 1, -1])

            var flattenedSumRowsY = magnitudesOp.Reshape(new int[] { 1, -1 }).Result;
            var flattenedSumRowsX = opWeights.Reshape(new int[] { 1, -1 }).Result;
            var flattenedSumRowsYOp = magnitudesOp.DeepClone();

            //flattened_Y_squared = flattened_sum_rows_Y * flattened_sum_rows_Y
            //flattened_X_squared = flattened_sum_rows_X * flattened_sum_rows_X

            var flattenedYSquared = magnitudesOp.Mul(flattenedSumRowsY).Result;
            var flattenedXSquared = opWeights.Mul(flattenedSumRowsX).Result;

            //added_YX_overall = flattened_Y_squared + flattened_X_squared
            var addedYXOverall = magnitudesOp.Add(flattenedXSquared).Result;

            //magnitudes_overall = tf.sqrt(added_YX_overall)
            //angles_overall = tf.atan2(flattened_sum_rows_Y, flattened_sum_rows_X)

            var magnitudesOverall = magnitudesOp.SquareRoot().Result;
            var anglesOverall = flattenedSumRowsYOp.Atan2(flattenedSumRowsX).Result;

            // reshaped_magnitudes_overall = tf.reshape(magnitudes_overall, [batch_size, rows, half_cols])
            // reshaped_angles_overall = tf.reshape(angles_overall, [batch_size, rows, half_cols])

            var reshapedMagnitudesOverall = magnitudesOp.Reshape(new int[] { rows, halfCols }).Result;
            var reshapedAnglesOverall = flattenedSumRowsYOp.Reshape(new int[] { rows, halfCols }).Result;

            //output_tensor = tf.concat([magnitudes_overall, angles_overall], axis = 2)
            //output_tensor = tf.reshape(output_tensor, [batch_size, rows, cols])

            var outputTensor = magnitudesOp.Concat(new[] { reshapedAnglesOverall }, axis: 1).Result;
            var output = magnitudesOp.Reshape(new int[] { rows, cols }).Result;

            var naiveOutput = resultTensor;

            var pradOpOutputCode = output.PrintCode();
            var naiveOutputCode = naiveOutput.PrintCode();

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

            var result = pradOp.CustomOperation(operation, reverseOperation, 1, new int[] { 2, 2 });

            Assert.Equal(new double[] { 1, 4, 9, 16 }, result.Result.Data);

            // Perform backpropagation
            var upstreamGradient = new Tensor(new int[] { 2, 2 }, new double[] { 1, 1, 1, 1 });
            pradOp.Back(upstreamGradient);

            Assert.Equal(new double[] { 2, 4, 6, 8 }, result.Gradients[0].Data);
        }
    }
}
