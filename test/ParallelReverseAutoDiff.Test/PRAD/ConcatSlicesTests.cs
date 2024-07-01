namespace ParallelReverseAutoDiff.Test.PRAD
{
    using Xunit;
    using System;
    using System.Linq;
    using ParallelReverseAutoDiff.PRAD;

    public class ConcatSlicesTests
    {
        [Fact]
        public void TestScenario1()
        {
            // Scenario 1: Input tensors: (9, 8, 7) and (2, 8, 7), axis range "1:3", concatAxis 0
            var tensor1 = CreateTensorWithUniqueValues(9, 8, 7);
            var tensor2 = CreateTensorWithUniqueValues(2, 8, 7);
            var tensors = new[] { tensor1, tensor2 };

            var result = Tensor.ConcatSlices(tensors, "1:3", 0);

            Assert.Equal(new int[] { 11, 8, 7 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);

            // Test reverse operation
            TensorReverse reverse = new TensorReverse(tensors);
            var gradients = reverse.ConcatSlicesReverse(result, "1:3", 0);

            Assert.Equal(2, gradients.Length);
            Assert.Equal(new int[] { 9, 8, 7 }, gradients[0].Shape);
            Assert.Equal(new int[] { 2, 8, 7 }, gradients[1].Shape);
            Assert.Equal(result.Data.Sum(), gradients[0].Data.Sum() + gradients[1].Data.Sum(), 6);

            // Verify that gradients match the original tensors
            Assert.True(TensorsEqual(tensor1, gradients[0]));
            Assert.True(TensorsEqual(tensor2, gradients[1]));
        }

        [Fact]
        public void TestScenario2()
        {
            // Scenario 2: Input tensors: (9, 8, 7) and (2, 8, 7), axis range "1:3", concatAxis 1
            var tensor1 = CreateTensorWithUniqueValues(9, 8, 7);
            var tensor2 = CreateTensorWithUniqueValues(2, 8, 7);
            var tensors = new[] { tensor1, tensor2 };

            var result = Tensor.ConcatSlices(tensors, "1:3", 1);

            Assert.Equal(new int[] { 1, 88, 7 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);

            // Test reverse operation
            TensorReverse reverse = new TensorReverse(tensors);
            var gradients = reverse.ConcatSlicesReverse(result, "1:3", 1);

            Assert.Equal(2, gradients.Length);
            Assert.Equal(new int[] { 9, 8, 7 }, gradients[0].Shape);
            Assert.Equal(new int[] { 2, 8, 7 }, gradients[1].Shape);
            Assert.Equal(result.Data.Sum(), gradients[0].Data.Sum() + gradients[1].Data.Sum(), 6);

            // Verify that gradients match the original tensors
            Assert.True(TensorsEqual(tensor1, gradients[0]));
            Assert.True(TensorsEqual(tensor2, gradients[1]));
        }

        [Fact]
        public void TestScenario3()
        {
            // Scenario 3: Input tensors: (9, 8, 7) and (2, 8, 7), axis range "1:3", concatAxis 2
            var tensor1 = CreateTensorWithUniqueValues(9, 8, 7);
            var tensor2 = CreateTensorWithUniqueValues(2, 8, 7);
            var tensors = new[] { tensor1, tensor2 };

            var result = Tensor.ConcatSlices(tensors, "1:3", 2);

            Assert.Equal(new int[] { 1, 8, 77 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);

            // Test reverse operation
            TensorReverse reverse = new TensorReverse(tensors);
            var gradients = reverse.ConcatSlicesReverse(result, "1:3", 2);

            Assert.Equal(2, gradients.Length);
            Assert.Equal(new int[] { 9, 8, 7 }, gradients[0].Shape);
            Assert.Equal(new int[] { 2, 8, 7 }, gradients[1].Shape);
            Assert.Equal(result.Data.Sum(), gradients[0].Data.Sum() + gradients[1].Data.Sum(), 6);

            // Verify that gradients match the original tensors
            Assert.True(TensorsEqual(tensor1, gradients[0]));
            Assert.True(TensorsEqual(tensor2, gradients[1]));
        }

        [Fact]
        public void TestScenario4()
        {
            // Scenario 4: Input tensors: (9, 8, 7) and (8, 7), axis range "-2:0", concatAxis 0
            var tensor1 = CreateTensorWithUniqueValues(9, 8, 7);
            var tensor2 = CreateTensorWithUniqueValues(8, 7);
            var tensors = new[] { tensor1, tensor2 };

            var result = Tensor.ConcatSlices(tensors, "-2:0", 0);

            Assert.Equal(new int[] { 10, 8, 7 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);

            // Test reverse operation
            TensorReverse reverse = new TensorReverse(tensors);
            var gradients = reverse.ConcatSlicesReverse(result, "-2:0", 0);

            Assert.Equal(2, gradients.Length);
            Assert.Equal(new int[] { 9, 8, 7 }, gradients[0].Shape);
            Assert.Equal(new int[] { 8, 7 }, gradients[1].Shape);
            Assert.Equal(result.Data.Sum(), gradients[0].Data.Sum() + gradients[1].Data.Sum(), 6);

            // Verify that gradients match the original tensors
            Assert.True(TensorsEqual(tensor1, gradients[0]));
            Assert.True(TensorsEqual(tensor2, gradients[1]));
        }

        [Fact]
        public void TestInvalidInput()
        {
            var tensor = CreateTensorWithUniqueValues(2, 2);
            var tensors = new[] { tensor };

            Assert.Throws<ArgumentException>(() => Tensor.ConcatSlices(null, "0:2", 0));
            Assert.Throws<ArgumentException>(() => Tensor.ConcatSlices(new Tensor[0], "0:2", 0));
            Assert.Throws<ArgumentException>(() => Tensor.ConcatSlices(tensors, "0:3", 0));
        }

        [Fact]
        public void TestConcatAxisEqualToRank()
        {
            var tensor1 = CreateTensorWithUniqueValues(2, 2);
            var tensor2 = CreateTensorWithUniqueValues(2, 2);
            var tensors = new[] { tensor1, tensor2 };

            // Test ConcatSlices
            var result = Tensor.ConcatSlices(tensors, "0:2", 2);
            Assert.Equal(new int[] { 1, 2, 4 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);

            // Test ConcatSlicesReverse
            TensorReverse reverse = new TensorReverse(tensors);
            var gradients = reverse.ConcatSlicesReverse(result, "0:2", 2);
            Assert.Equal(2, gradients.Length);
            Assert.Equal(new int[] { 2, 2 }, gradients[0].Shape);
            Assert.Equal(new int[] { 2, 2 }, gradients[1].Shape);
            Assert.Equal(result.Data.Sum(), gradients[0].Data.Sum() + gradients[1].Data.Sum(), 6);

            // Verify that gradients match the original tensors
            Assert.True(TensorsEqual(tensor1, gradients[0]));
            Assert.True(TensorsEqual(tensor2, gradients[1]));
        }

        private Tensor CreateTensorWithUniqueValues(params int[] shape)
        {
            var tensor = new Tensor(shape);
            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = i + 1;
            }
            return tensor;
        }

        private bool TensorsEqual(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                return false;

            for (int i = 0; i < a.Data.Length; i++)
            {
                if (Math.Abs(a.Data[i] - b.Data[i]) > 1e-6)
                    return false;
            }

            return true;
        }
    }
}
