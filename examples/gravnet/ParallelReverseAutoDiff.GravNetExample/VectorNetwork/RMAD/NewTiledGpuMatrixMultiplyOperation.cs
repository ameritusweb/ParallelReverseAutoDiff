//------------------------------------------------------------------------------
// <copyright file="NewTiledGpuMatrixMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using ILGPU;
    using ILGPU.Runtime;
    using ParallelReverseAutoDiff.Exceptions;
    using ParallelReverseAutoDiff.GravNetExample.Common;

    /// <summary>
    /// GPU Tiled matrix multiplication operation.
    /// </summary>
    public class NewTiledGpuMatrixMultiplyOperation : Operation
    {
        private const int TILESIZE = 8;
        private Matrix[,] input1;
        private Matrix[,] input2;
        private Matrix[,] output;
        private Matrix[,] dInput1;
        private Matrix[,] dInput2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new NewTiledGpuMatrixMultiplyOperation();
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

            var total = 0.0d; // Initialize accumulator for sums across tiles

            for (var i = 0; i < aView.IntExtent.Y; i += TILESIZE)
            {
                var sum = 0.0d;

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

                var kk = 0;

                for (var k = 0; k < TILESIZE; k++)
                {
                    sum += aTile[new Index2D(x, k)] * bTile[new Index2D(k, y)];
                }

                Group.Barrier();

                total += sum;
            }

            if (global.X < cView.IntExtent.X && global.Y < cView.IntExtent.Y)
            {
                cView[global] = total;
            }
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply function.
        /// </summary>
        /// <param name="input1">The first input to the matrix multiply operation.</param>
        /// <param name="input2">The second input to the matrix multiply operation.</param>
        /// <returns>The output of the matrix multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            var brokenInput1 = CommonMatrixUtils.BreakIntoSections(input1, 8);
            var brokenInput2 = CommonMatrixUtils.BreakIntoSections(input2, 8);

            this.input1 = new Matrix[brokenInput1.GetLength(0), brokenInput1.GetLength(1)];
            this.input2 = new Matrix[brokenInput2.GetLength(0), brokenInput2.GetLength(1)];
            this.output = new Matrix[brokenInput1.GetLength(0), brokenInput2.GetLength(1)];

            Parallel.For(0, 8, i =>
            {
                for (int j = 0; j < 8; j++)
                {
                    var i1 = brokenInput1[i, j];
                    var i2 = brokenInput2[i, j];

                    this.InnerForward(i, j, i1, i2);
                }
            });

            this.Output = CommonMatrixUtils.PieceTogether(this.output);
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            this.dInput1 = new Matrix[this.input1.GetLength(0), this.input1.GetLength(1)];
            this.dInput2 = new Matrix[this.input2.GetLength(0), this.input2.GetLength(1)];
            var dOutputSections = CommonMatrixUtils.BreakIntoSections(dOutput, 8);

            Parallel.For(0, this.dInput1.GetLength(0), i =>
            {
                for (int j = 0; j < this.dInput2.GetLength(1); j++)
                {
                    this.InnerBackward(i, j, dOutputSections[i, j]);
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(CommonMatrixUtils.PieceTogether(this.dInput1))
                .AddInputGradient(CommonMatrixUtils.PieceTogether(this.dInput2))
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

        private void InnerForward(int ii, int jj, Matrix input1, Matrix input2)
        {
            this.input1[ii, jj] = input1;
            this.input2[ii, jj] = input2;
            int input1Cols = input1[0].Length;
            int input2Rows = input2.Length;

            if (input1Cols != input2Rows)
            {
                throw new InvalidOperationException("Input 1 columns do not match Input 2 rows");
            }

            var acceleratedTiledResult = this.MatrixMultiplyTiled(CudaBlas.Instance.Accelerator, this.To2D(input1.ToArray(), false), this.To2D(input2.ToArray(), false));

            this.output[ii, jj] = new Matrix(this.ToJagged(acceleratedTiledResult));
        }

        private void InnerBackward(int ii, int jj, Matrix dOutput)
        {
            // Calculate gradient w.r.t. input1

            // Compute dInput1 using MatrixMultiply
            var acceleratedTiledResult1 = this.MatrixMultiplyTiled(CudaBlas.Instance.Accelerator, this.To2D(dOutput.ToArray(), false), this.To2D(this.input2[ii, jj].ToArray(), true));
            this.dInput1[ii, jj] = new Matrix(this.ToJagged(acceleratedTiledResult1));

            // Calculate gradient w.r.t. input2

            // Compute dInput2 using MatrixMultiply
            var acceleratedTiledResult2 = this.MatrixMultiplyTiled(CudaBlas.Instance.Accelerator, this.To2D(this.input1[ii, jj].ToArray(), true), this.To2D(dOutput.ToArray(), false));
            this.dInput2[ii, jj] = new Matrix(this.ToJagged(acceleratedTiledResult2));
        }

        /// <summary>
        /// Converts a jagged array to a 2D array.
        /// </summary>
        /// <param name="source">The jagged array.</param>
        /// <param name="transpose">Whether to transpose the array.</param>
        /// <returns>The 2-D array.</returns>
        private double[,] To2D(double[][] source, bool transpose)
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
        private double[][] ToJagged(double[,] source)
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
