//------------------------------------------------------------------------------
// <copyright file="CudaBlas.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Threading;
    using System.Threading.Tasks;
    using System.Xml;
    using ManagedCuda;
    using ParallelReverseAutoDiff.Interprocess;
    using Cuda = ManagedCuda.CudaBlas;

    /// <summary>
    /// Wrapper for CUBLAS library.
    /// </summary>
    public class CudaBlas : IDisposable
    {
        private static readonly Lazy<CudaBlas> LazyLoadedInstance = new Lazy<CudaBlas>(() => new CudaBlas(), true);

        private readonly CircularBuffer circularBuffer;

        private readonly ConcurrentDictionary<long, Matrix> resultDictionary;

        private Semaphore producerMutex;

        private Semaphore consumerMutex;

        private Semaphore resultMutex;

        private Semaphore initializationMutex;

        private PrimaryContext primaryContext;

        private ManagedCuda.CudaBlas.CudaBlas blas;

        private CudaDeviceVariable<double> dA;

        private CudaDeviceVariable<double> dB;

        private CudaDeviceVariable<double> dC;

        private bool isInitialized;

        private bool areDeviceVariablesInitialized;

        private CudaBlas()
        {
            this.initializationMutex = new Semaphore(0, 1);
            this.circularBuffer = new CircularBuffer(1024 * 1024 * 8);
            this.resultDictionary = new ConcurrentDictionary<long, Matrix>();
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
            Task.Delay(1).ContinueWith((t) =>
            {
                PrimaryContext ctx = new PrimaryContext(this.DeviceId);
                this.primaryContext = ctx;
                this.blas = new ManagedCuda.CudaBlas.CudaBlas(ManagedCuda.CudaBlas.PointerMode.Host);
                this.producerMutex = new Semaphore(0, 1);
                this.consumerMutex = new Semaphore(1, 1);
                this.resultMutex = new Semaphore(0, 1);
                this.isInitialized = true;
                this.initializationMutex.Release();
                this.CudaThreadMethod();
            });
            this.initializationMutex.WaitOne();
        }

        /// <summary>
        /// Disposes the CUBLAS library.
        /// </summary>
        public void Dispose()
        {
            if (this.areDeviceVariablesInitialized)
            {
                this.DisposeDeviceVariables();
            }

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
        /// <param name="transposeMatrix1">Whether to transpose matrix A before multiplying.</param>
        /// <param name="transposeMatrix2">Whether to transpose matrix B before multiplying.</param>
        public void InitializeDeviceVariables(int rows1, int cols1, int rows2, int cols2, bool transposeMatrix1, bool transposeMatrix2)
        {
            int m1 = transposeMatrix1 ? cols1 : rows1;
            int m2 = transposeMatrix2 ? rows2 : cols2;
            int c = m1 * m2;
            if (!this.AreDeviceVariablesInitialized)
            {
                this.dA = new CudaDeviceVariable<double>(rows1 * cols1);
                this.dB = new CudaDeviceVariable<double>(rows2 * cols2);
                this.dC = new CudaDeviceVariable<double>(c);
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

                if (this.dC.Size != c)
                {
                    this.dC.Dispose();
                    this.dC = new CudaDeviceVariable<double>(c);
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

            int c1 = transposeMatrix1 ? n : m;
            int c2 = transposeMatrix2 ? p : k;

            this.InitializeDeviceVariables(m, n, p, k, transposeMatrix1, transposeMatrix2);

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
                c = MatrixUtils.ReshapeMatrix(c_flat, c1, c2);
            }
            catch (Exception)
            {
                this.DisposeDeviceVariables();
                this.Dispose();
                throw new InvalidOperationException("CUBLAS failure.");
            }

            return c;
        }

        /// <summary>
        /// Performs a matrix multiplication using the CUBLAS library.
        /// </summary>
        /// <param name="matrixFlat1">Matric A.</param>
        /// <param name="rows1">The number of rows in the first matrix.</param>
        /// <param name="cols1">The number of columns in the first matrix.</param>
        /// <param name="transposeMatrix1">Whether to transpose matrix A before multiplying.</param>
        /// <param name="matrixFlat2">Matrix B.</param>
        /// <param name="rows2">The number of rows in the second matrix.</param>
        /// <param name="cols2">The number of columns in the second matrix.</param>
        /// <param name="transposeMatrix2">Whether to transpose matrix B before multiplying.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix MatrixMultiply(double[] matrixFlat1, int rows1, int cols1, bool transposeMatrix1, double[] matrixFlat2, int rows2, int cols2, bool transposeMatrix2)
        {
            int m = transposeMatrix1 ? cols1 : rows1;
            int n = transposeMatrix2 ? rows2 : cols2;
            int k = transposeMatrix1 ? rows1 : cols1;

            int c1 = m;
            int c2 = n;

            this.InitializeDeviceVariables(rows1, cols1, rows2, cols2, transposeMatrix1, transposeMatrix2);

            this.dA.CopyToDevice(matrixFlat1);
            this.dB.CopyToDevice(matrixFlat2);

            Matrix? c;
            try
            {
                // Call Gemm to perform the matrix multiplication
                double alpha = 1.0;
                double beta = 0.0;
                this.blas.Gemm(transposeMatrix1 ? Cuda.Operation.Transpose : Cuda.Operation.NonTranspose, transposeMatrix2 ? Cuda.Operation.Transpose : Cuda.Operation.NonTranspose, m, n, k, alpha, this.dA, rows1, this.dB, rows2, beta, this.dC, c1);

                // Copy the result back to the host and reshape it to a 2D array
                double[] c_flat = this.dC;
                c = MatrixUtils.ReshapeMatrix(c_flat, c1, c2);
            }
            catch (Exception)
            {
                this.DisposeDeviceVariables();
                this.Dispose();
                throw new InvalidOperationException("CUBLAS failure.");
            }

            return c;
        }

        /// <summary>
        /// Writes the matrices to shared memory.
        /// </summary>
        /// <param name="matrixA">Matrix A.</param>
        /// <param name="transposeA">Whether to transpose matrix A before multiplying.</param>
        /// <param name="matrixB">Matrix B.</param>
        /// <param name="transposeB">Whether to transpose matrix B before multiplying.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix WriteMatricesToSharedMemory(Matrix matrixA, bool transposeA, Matrix matrixB, bool transposeB)
        {
            this.consumerMutex.WaitOne();

            int bufferSizeA = 1 + (3 * sizeof(int)) + (matrixA.Rows * matrixA.Cols * sizeof(double));
            int bufferSizeB = 1 + (3 * sizeof(int)) + (matrixB.Rows * matrixB.Cols * sizeof(double));
            int totalBufferSize = bufferSizeA + bufferSizeB;

            // Serialize the matrices into a buffer
            byte[] serializedMatrices = new byte[totalBufferSize];

            // Serialize matrixA
            matrixA.Serialize(serializedMatrices.AsSpan(0, bufferSizeA), transposeA);

            // Serialize matrixB
            matrixB.Serialize(serializedMatrices.AsSpan(bufferSizeA, bufferSizeB), transposeB);

            if (serializedMatrices.Length > this.circularBuffer.Capacity)
            {
                this.circularBuffer.Resize(serializedMatrices.Length);
            }

            // Write the serialized matrices to the circular buffer
            this.circularBuffer.Write(serializedMatrices, 0);

            this.producerMutex.Release();
            this.resultMutex.WaitOne();

            Matrix result;
            this.resultDictionary.TryGetValue(matrixA.UniqueId * matrixB.UniqueId, out result);
            return result;
        }

        /// <summary>
        /// A method that runs on a separate thread and performs the matrix multiplication using CudaBlas.
        /// </summary>
        public void CudaThreadMethod()
        {
            while (true)
            {
                // Wait for the producer to signal that data is available
                this.producerMutex.WaitOne();

                // Read the transpose flags, unique IDs, rows, and columns of the matrices
                byte transposeFlagA = this.circularBuffer.Read(0, 1).Span[0];
                int uniqueIDA = BitConverter.ToInt32(this.circularBuffer.Read(1, sizeof(int)).Span);
                int rowsA = BitConverter.ToInt32(this.circularBuffer.Read(1 + sizeof(int), sizeof(int)).Span);
                int colsA = BitConverter.ToInt32(this.circularBuffer.Read(1 + (2 * sizeof(int)), sizeof(int)).Span);

                // Calculate the serialized length of the first matrix
                int matrixASerializedLength = 1 + (3 * sizeof(int)) + (rowsA * colsA * sizeof(double));

                byte transposeFlagB = this.circularBuffer.Read(matrixASerializedLength, 1).Span[0];
                int uniqueIDB = BitConverter.ToInt32(this.circularBuffer.Read(matrixASerializedLength + 1, sizeof(int)).Span);
                int rowsB = BitConverter.ToInt32(this.circularBuffer.Read(matrixASerializedLength + 1 + sizeof(int), sizeof(int)).Span);
                int colsB = BitConverter.ToInt32(this.circularBuffer.Read(matrixASerializedLength + 1 + (2 * sizeof(int)), sizeof(int)).Span);

                if (rowsA > 0 && rowsB > 0 && colsA > 0 && colsB > 0)
                {
                    // Calculate the serialized length of the second matrix
                    int matrixBSerializedLength = 1 + (3 * sizeof(int)) + (rowsB * colsB * sizeof(double));

                    // Read the serialized matrices from the circular buffer
                    ReadOnlyMemory<byte> serializedMatrixA = this.circularBuffer.Read(0, matrixASerializedLength);
                    ReadOnlyMemory<byte> serializedMatrixB = this.circularBuffer.Read(matrixASerializedLength, matrixBSerializedLength);

                    // Deserialize the flat matrices
                    double[] matrixFlatA = Matrix.DeserializeToFlatArray(serializedMatrixA.Span);
                    double[] matrixFlatB = Matrix.DeserializeToFlatArray(serializedMatrixB.Span);

                    // Perform the matrix multiplication using CudaBlas
                    Matrix result = this.MatrixMultiply(matrixFlatA, rowsA, colsA, transposeFlagA == 1, matrixFlatB, rowsB, colsB, transposeFlagB == 1);

                    // Store the result in the concurrent dictionary with the unique identifier
                    this.resultDictionary.TryAdd(uniqueIDA * uniqueIDB, result);
                }

                // Signal the producer that the consumer has finished processing
                this.resultMutex.Release();
                this.consumerMutex.Release();

                if (this.isInitialized == false)
                {
                    break;
                }
            }
        }
    }
}
