//------------------------------------------------------------------------------
// <copyright file="CudaBlas.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using ManagedCuda;
    using Cuda = ManagedCuda.CudaBlas;

    /// <summary>
    /// Wrapper for CUBLAS library.
    /// </summary>
    public class CudaBlas : IDisposable
    {
        private static readonly Lazy<CudaBlas> LazyLoadedInstance = new Lazy<CudaBlas>(() => new CudaBlas(), true);

        private PrimaryContext primaryContext;

        private ManagedCuda.CudaBlas.CudaBlas blas;

        private CudaDeviceVariable<double> dA;

        private CudaDeviceVariable<double> dB;

        private CudaDeviceVariable<double> dC;

        private bool isInitialized;

        private bool areDeviceVariablesInitialized;

        private CudaBlas()
        {
        }

        /// <summary>
        /// Gets the singleton instance of the CUBLAS library.
        /// </summary>
        public static CudaBlas Instance
        {
            get
            {
                return LazyLoadedInstance.Value;
            }
        }

        /// <summary>
        /// Gets a value indicating whether the CUBLAS library is initialized.
        /// </summary>
        public bool IsInitialized
        {
            get
            {
                return this.isInitialized;
            }
        }

        /// <summary>
        /// Gets a value indicating whether the device variables are initialized.
        /// </summary>
        public bool AreDeviceVariablesInitialized
        {
            get
            {
                return this.areDeviceVariablesInitialized;
            }
        }

        /// <summary>
        /// Gets or sets the DeviceId for the GPU to use.
        /// </summary>
        public int DeviceId { get; set; }

        /// <summary>
        /// Initializes the CUBLAS library.
        /// </summary>
        public void Initialize()
        {
            PrimaryContext ctx = new PrimaryContext(this.DeviceId);
            this.primaryContext = ctx;
            this.blas = new ManagedCuda.CudaBlas.CudaBlas(ManagedCuda.CudaBlas.PointerMode.Host);
            this.isInitialized = true;
        }

        /// <summary>
        /// Disposes the CUBLAS library.
        /// </summary>
        public void Dispose()
        {
            this.blas.Dispose();
            this.primaryContext.Dispose();
            this.isInitialized = false;
        }

        /// <summary>
        /// Initializes the device variables.
        /// </summary>
        /// <param name="rows1">The number of rows in the first matrix.</param>
        /// <param name="cols1">The number of columns in the first matrix.</param>
        /// <param name="rows2">The number of rows in the second matrix.</param>
        /// <param name="cols2">The number of columns in the second matrix.</param>
        public void InitializeDeviceVariables(int rows1, int cols1, int rows2, int cols2)
        {
            if (!this.AreDeviceVariablesInitialized)
            {
                this.dA = new CudaDeviceVariable<double>(rows1 * cols1);
                this.dB = new CudaDeviceVariable<double>(rows2 * cols2);
                this.dC = new CudaDeviceVariable<double>(rows1 * cols2);
            }
            else
            {
                if (this.dA.Size != rows1 * cols1)
                {
                    this.dA.Dispose();
                    this.dA = new CudaDeviceVariable<double>(rows1 * cols1);
                }

                if (this.dB.Size != rows2 * cols2)
                {
                    this.dB.Dispose();
                    this.dB = new CudaDeviceVariable<double>(rows2 * cols2);
                }

                if (this.dC.Size != rows1 * cols2)
                {
                    this.dC.Dispose();
                    this.dC = new CudaDeviceVariable<double>(rows1 * cols2);
                }
            }

            this.areDeviceVariablesInitialized = true;
        }

        /// <summary>
        /// Disposes the device variables.
        /// </summary>
        public void DisposeDeviceVariables()
        {
            this.dA.Dispose();
            this.dB.Dispose();
            this.dC.Dispose();
            this.areDeviceVariablesInitialized = false;
        }

        /// <summary>
        /// Performs a matrix multiplication using the CUBLAS library.
        /// </summary>
        /// <param name="matrix1">Matric A.</param>
        /// <param name="transposeMatrix1">Whether to transpose matrix A before multiplying.</param>
        /// <param name="matrix2">Matrix B.</param>
        /// <param name="transposeMatrix2">Whether to transpose matrix B before multiplying.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix MatrixMultiply(Matrix matrix1, bool transposeMatrix1, Matrix matrix2, bool transposeMatrix2)
        {
            int m = matrix1.Length;
            int n = matrix1[0].Length;
            int p = matrix2.Length;
            int k = matrix2[0].Length;

            this.InitializeDeviceVariables(m, n, p, k);

            // Convert input matrices to 1D arrays and create CudaDeviceVariable objects
            double[] a_flat = MatrixUtils.FlattenMatrix(matrix1);
            double[] b_flat = MatrixUtils.FlattenMatrix(matrix2);

            this.dA.CopyToDevice(a_flat);
            this.dB.CopyToDevice(b_flat);

            Matrix? c;
            try
            {
                // Call Gemm to perform the matrix multiplication
                double alpha = 1.0;
                double beta = 0.0;
                this.blas.Gemm(transposeMatrix1 ? Cuda.Operation.Transpose : Cuda.Operation.NonTranspose, transposeMatrix2 ? Cuda.Operation.Transpose : Cuda.Operation.NonTranspose, m, n, k, alpha, this.dA, m, this.dB, k, beta, this.dC, m);

                // Copy the result back to the host and reshape it to a 2D array
                double[] c_flat = this.dC;
                c = MatrixUtils.ReshapeMatrix(c_flat, m, k);
            }
            catch (Exception)
            {
                this.DisposeDeviceVariables();
                this.Dispose();
                throw new InvalidOperationException("CUBLAS failure.");
            }

            return c;
        }
    }
}
