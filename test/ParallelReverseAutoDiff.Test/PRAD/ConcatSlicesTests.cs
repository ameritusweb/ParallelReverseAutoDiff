﻿namespace ParallelReverseAutoDiff.Test.PRAD
{
    using Xunit;
    using System;
    using System.Linq;
    using ParallelReverseAutoDiff.PRAD;
    using System.Diagnostics;

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
        }

        [Fact]
        public void TestScenario5()
        {
            // Scenario 4: Input tensors: (9, 8, 7) and (8, 7), axis range "-2:0", concatAxis 1
            var tensor1 = CreateTensorWithUniqueValues(9, 8, 7);
            var tensor2 = CreateTensorWithUniqueValues(8, 7);
            var tensors = new[] { tensor1, tensor2 };

            var result = Tensor.ConcatSlices(tensors, "-2:0", 1);

            Assert.Equal(new int[] { 1, 80, 7 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);
        }

        [Fact]
        public void TestScenario6()
        {
            // Scenario 4: Input tensors: (9, 8, 7) and (8, 7), axis range "-2:0", concatAxis 2
            var tensor1 = CreateTensorWithUniqueValues(9, 8, 7);
            var tensor2 = CreateTensorWithUniqueValues(8, 7);
            var tensors = new[] { tensor1, tensor2 };

            var result = Tensor.ConcatSlices(tensors, "-2:0", 2);

            Assert.Equal(new int[] { 1, 8, 70 }, result.Shape);
            Assert.Equal(tensor1.Data.Sum() + tensor2.Data.Sum(), result.Data.Sum(), 6);
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

        private void PrintTensor(Tensor tensor)
        {
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    if (tensor.Shape.Length == 3)
                    {
                        for (int k = 0; k < tensor.Shape[2]; k++)
                        {
                            Debug.Write($"{tensor[i, j, k]} ");
                        }
                    }
                    else
                    {
                        Debug.Write($"{tensor[i, j]} ");
                    }
                }
                Debug.WriteLine("");
            }
            Debug.WriteLine("");
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