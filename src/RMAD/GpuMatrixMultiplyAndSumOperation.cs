//------------------------------------------------------------------------------
// <copyright file="GpuMatrixMultiplyAndSumOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using ILGPU;
    using ILGPU.Runtime;

    /// <summary>
    /// GPU matrix multiplication and summation operation.
    /// </summary>
    public class GpuMatrixMultiplyAndSumOperation : Operation
    {
        private const int TILESIZE = 8;
        private Matrix a;
        private DeepMatrix b;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new GpuMatrixMultiplyAndSumOperation();
        }

        /// <summary>
        /// The tiled matrix multiplication kernel that runs on the accelerated device.
        /// </summary>
        /// <param name="aView">An input matrix of size MxK.</param>
        /// <param name="bView">An input matrix of size KxN.</param>
        /// <param name="cView">An output matrix of size MxN.</param>
        public static void MatrixMultiplyTiledKernel(
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView,
            ArrayView2D<double, Stride2D.DenseX> cView)
        {
            var global = Grid.GlobalIndex.XY;
            var x = Group.IdxX;
            var y = Group.IdxY;

            var aTile = SharedMemory.Allocate2D<double, Stride2D.DenseX>(new Index2D(TILESIZE, TILESIZE), new Stride2D.DenseX(TILESIZE));
            var bTile = SharedMemory.Allocate2D<double, Stride2D.DenseX>(new Index2D(TILESIZE, TILESIZE), new Stride2D.DenseX(TILESIZE));
            var sum = 0.0d;

            for (var i = 0; i < aView.IntExtent.X; i += TILESIZE)
            {
                if (global.X < aView.IntExtent.X && y + i < aView.IntExtent.Y)
                {
                    aTile[x, y] = aView[global.X, y + i];
                }
                else
                {
                    aTile[x, y] = 0;
                }

                if (x + i < bView.IntExtent.X && global.Y < bView.IntExtent.Y)
                {
                    bTile[x, y] = bView[x + i, global.Y];
                }
                else
                {
                    bTile[x, y] = 0;
                }

                Group.Barrier();

                for (var k = 0; k < TILESIZE; k++)
                {
                    sum += aTile[new Index2D(x, k)] * bTile[new Index2D(k, y)];
                }

                Group.Barrier();
            }

            if (global.X < cView.IntExtent.X && global.Y < cView.IntExtent.Y)
            {
                cView[global] = sum;
            }
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            // Store the intermediate matrices
            this.IntermediateMatrices.AddOrUpdate(id, this.a, (x, y) => this.a);
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.b, (x, y) => this.b);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            // Restore the intermediate matrices
            this.a = this.IntermediateMatrices[id];
            this.b = this.IntermediateDeepMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="a">NxN Matrix.</param>
        /// <param name="b">PxNxN DeepMatrix.</param>
        /// <returns>1xP Matrix.</returns>
        public Matrix Forward(Matrix a, DeepMatrix b)
        {
            this.a = a;
            this.b = b;

            int p = b.Depth;
            this.Output = new Matrix(1, p);

            for (int q = 0; q < p; q++)
            {
                Matrix slice = b[q];
                var acceleratedTiledResult = this.MatrixMultiplyTiled(CudaBlas.Instance.Accelerator, this.To2D(a.ToArray(), false), this.To2D(slice.ToArray(), false));
                Matrix matrixMultiplied = new Matrix(this.ToJagged(acceleratedTiledResult));
                double sum = matrixMultiplied.Sum();
                this.Output[0][q] = sum;
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int p = this.b.Depth;
            DeepMatrix dInputB = new DeepMatrix(p, this.a.Rows, this.a.Cols);

            Matrix dInputA = new Matrix(this.a.Rows, this.a.Cols);

            for (int q = 0; q < p; q++)
            {
                Matrix dSlice = this.a.Transpose() * dOutput[0][q]; // Multiply by scalar and then by matrix
                dInputB[q] = dSlice;

                dInputA = dInputA + (this.b[q].Transpose() * dOutput[0][q]); // Multiply by scalar and then by matrix
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputA)
                .AddInputGradientArray(dInputB)
                .Build();
        }

        /// <summary>
        /// Multiplies two dense matrices and returns the resultant matrix (using tiling).
        /// </summary>
        /// <param name="accelerator">The Accelerator to run the multiplication on.</param>
        /// <param name="a">A dense MxK matrix.</param>
        /// <param name="b">A dense KxN matrix.</param>
        /// <returns>A dense MxN matrix.</returns>
        public double[,] MatrixMultiplyTiled(Accelerator accelerator, double[,] a, double[,] b)
        {
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            if (ka != kb)
            {
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));
            }

            var kernel = accelerator.LoadStreamKernel<
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>>(
                MatrixMultiplyTiledKernel);
            var groupSize = new Index2D(TILESIZE, TILESIZE);
            var numGroups = new Index2D((m + TILESIZE - 1) / TILESIZE, (n + TILESIZE - 1) / TILESIZE);

            using var aBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, ka));
            using var bBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(ka, n));
            using var cBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, n));
            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);

            kernel((numGroups, groupSize), aBuffer, bBuffer, cBuffer);

            // Reads data from the GPU buffer into a new CPU array.
            // Implicitly calls accelerator.DefaultStream.Synchronize() to ensure
            // that the kernel and memory copy are completed first.
            return cBuffer.GetAsArray2D();
        }

        /// <summary>
        /// Converts a jagged array to a 2D array.
        /// </summary>
        /// <param name="source">The jagged array.</param>
        /// <param name="transpose">Whether to transpose the array.</param>
        /// <returns>The 2-D array.</returns>
        public double[,] To2D(double[][] source, bool transpose)
        {
            try
            {
                int firstDim = source.Length;
                int secondDim = source.GroupBy(row => row.Length).Single().Key; // throws InvalidOperationException if source is not rectangular

                if (transpose)
                {
                    var result = new double[secondDim, firstDim];
                    for (int i = 0; i < secondDim; ++i)
                    {
                        for (int j = 0; j < firstDim; ++j)
                        {
                            result[i, j] = source[j][i];
                        }
                    }

                    return result;
                }
                else
                {
                    var result = new double[firstDim, secondDim];
                    for (int i = 0; i < firstDim; ++i)
                    {
                        for (int j = 0; j < secondDim; ++j)
                        {
                            result[i, j] = source[i][j];
                        }
                    }

                    return result;
                }
            }
            catch (InvalidOperationException)
            {
                throw new InvalidOperationException("The given jagged array is not rectangular.");
            }
        }

        /// <summary>
        /// Converts a 2D array to a jagged array.
        /// </summary>
        /// <param name="source">The 2-D array.</param>
        /// <returns>The jagged array.</returns>
        public double[][] ToJagged(double[,] source)
        {
            int firstDim = source.GetLength(0);
            int secondDim = source.GetLength(1);
            var result = new double[firstDim][];

            for (int i = 0; i < firstDim; ++i)
            {
                result[i] = new double[secondDim];
                for (int j = 0; j < secondDim; ++j)
                {
                    result[i][j] = source[i, j];
                }
            }

            return result;
        }
    }
}