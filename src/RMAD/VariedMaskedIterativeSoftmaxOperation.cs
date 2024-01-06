//------------------------------------------------------------------------------
// <copyright file="VariedMaskedIterativeSoftmaxOperation.cs" author="ameritusweb" date="12/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using ILGPU;
    using ILGPU.Runtime;

    /// <summary>
    /// Varied masked iterative softmax operation.
    /// </summary>
    public class VariedMaskedIterativeSoftmaxOperation : Operation
    {
        private double temperature;
        private double maskThreshold;
        private Matrix input;
        private List<double[]> softmaxIterations;
        private List<double[]> previousSoftmaxOutputIterations;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new VariedMaskedIterativeSoftmaxOperation();
        }

        /// <summary>
        /// The gradient kernel.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="softmaxView">The softmax view.</param>
        /// <param name="previousOutputView">The previous output view.</param>
        /// <param name="currentdLdOutputView">The current output view.</param>
        /// <param name="dXView">The dx view.</param>
        /// <param name="dTempView">The dtemp view.</param>
        /// <param name="maskThreshold">The mask threshold.</param>
        /// <param name="temperature">The temperature.</param>
        /// <param name="inputView">The input view.</param>
        /// <param name="numCols">The num cols.</param>
        public static void GradientKernel(
            Index1D index,
            ArrayView<double> softmaxView,
            ArrayView<double> previousOutputView,
            ArrayView<double> currentdLdOutputView,
            ArrayView<double> dXView,
            ArrayView<double> dTempView,
            double maskThreshold,
            double temperature,
            ArrayView<double> inputView,
            int numCols)
        {
            if (index.X < numCols)
            {
                var i = index.X;
                if (previousOutputView[i] <= maskThreshold)
                {
                    double dXValue = 0.0;
                    double dTempValue = 0.0;

                    for (int j = 0; j < numCols; j++)
                    {
                        double softmaxGrad = softmaxView[i] * ((i == j ? 1.0 : 0.0) - softmaxView[j]);
                        double dSoftmax_dX = softmaxGrad / temperature;
                        double dSoftmax_dTemp = -inputView[i] * softmaxGrad / Math.Pow(temperature, 2);

                        dXValue += dSoftmax_dX * currentdLdOutputView[j];
                        dTempValue += dSoftmax_dTemp * currentdLdOutputView[j];
                    }

                    dXView[i] = dXValue;
                    dTempView[i] = dTempValue;
                }
            }
        }

        /// <summary>
        /// Performs the forward operation for the softmax function with temperature scaling and masking.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <param name="temp">The temperature to use for the softmax operation.</param>
        /// <param name="previousSoftmaxOutput">The previous softmax output used for masking.</param>
        /// <param name="maskThreshold">The threshold above which values in the previous output are masked.</param>
        /// <returns>The output of the softmax operation with masking.</returns>
        public Matrix Forward(Matrix input, Matrix temp, Matrix previousSoftmaxOutput, double maskThreshold)
        {
            // Initialization
            this.input = input;
            this.temperature = temp[0].Sum();
            this.maskThreshold = maskThreshold;
            this.softmaxIterations = new List<double[]>();
            this.previousSoftmaxOutputIterations = new List<double[]>();

            if (!previousSoftmaxOutput[0].Any(value => value > maskThreshold))
            {
                this.Output = (Matrix)input.Clone();
                return this.Output;
            }

            // Matrix to store values above the mask threshold from the initial input and each iteration
            Matrix storedValues = new Matrix(1, input.Cols);

            // Populate storedValues with values above the mask threshold from previousSoftmaxOutput
            for (int i = 0; i < previousSoftmaxOutput.Cols; i++)
            {
                storedValues[0][i] = previousSoftmaxOutput[0][i] > maskThreshold ? previousSoftmaxOutput[0][i] : 0;
            }

            // Loop until all values are masked out or below the mask threshold
            while (previousSoftmaxOutput[0].Any(value => value > maskThreshold && value < double.MaxValue))
            {
                double sumExp = 0;
                double[] expValues = new double[this.input.Cols];

                // Compute exp values and sumExp
                for (int i = 0; i < this.input.Cols; i++)
                {
                    if (previousSoftmaxOutput[0][i] <= maskThreshold)
                    {
                        expValues[i] = Math.Exp(this.input[0][i] / this.temperature);
                        sumExp += expValues[i];
                    }
                }

                this.softmaxIterations.Add(expValues);
                this.previousSoftmaxOutputIterations.Add(previousSoftmaxOutput[0].ToArray());

                // Compute softmax and update previousSoftmaxOutput
                for (int i = 0; i < this.input.Cols; i++)
                {
                    if (previousSoftmaxOutput[0][i] <= maskThreshold)
                    {
                        double softmaxValue = expValues[i] / sumExp;
                        previousSoftmaxOutput[0][i] = softmaxValue;

                        // Update storedValues if current softmax value is above the threshold
                        if (softmaxValue > maskThreshold)
                        {
                            storedValues[0][i] = softmaxValue;
                        }
                    }
                    else
                    {
                        // Mask the value by setting it to max value
                        previousSoftmaxOutput[0][i] = double.MaxValue;
                    }
                }
            }

            this.Output = new Matrix(storedValues[0].ToArray());

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            var accelerator = CudaBlas.Instance.Accelerator; // loads the CUDA accelerator

            int numCols = this.input.Cols;
            Matrix dX = new Matrix(1, numCols);
            Matrix dTemp = new Matrix(1, numCols);

            if (this.softmaxIterations.Count == 0)
            {
                return new BackwardResultBuilder()
                    .AddInputGradient(dLdOutput)
                    .AddInputGradient(dTemp)
                    .Build();
            }

            Matrix dXParallel = new Matrix(this.softmaxIterations.Count, numCols);
            Matrix dTempParallel = new Matrix(this.softmaxIterations.Count, numCols);

            // Allocate memory outside the loop
            using var dSoftmaxGpu = accelerator.Allocate1D<double>(numCols);
            using var dPreviousOutputGpu = accelerator.Allocate1D<double>(numCols);
            using var dCurrentdLdOutputGpu = accelerator.Allocate1D<double>(numCols);
            using var dXGpu = accelerator.Allocate1D<double>(numCols);
            using var dTempGpu = accelerator.Allocate1D<double>(numCols);
            using var dInputGpu = accelerator.Allocate1D<double>(numCols);
            dInputGpu.CopyFromCPU(this.input[0]); // Assuming this.input[0] doesn't change

            // Iterate over stored softmax and previousSoftmaxOutput values
            for (int iteration = 0; iteration < this.softmaxIterations.Count; iteration++)
            {
                double[] softmax = this.softmaxIterations[iteration];
                double[] previousOutput = this.previousSoftmaxOutputIterations[iteration];

                dSoftmaxGpu.CopyFromCPU(softmax);
                dPreviousOutputGpu.CopyFromCPU(previousOutput);
                dCurrentdLdOutputGpu.CopyFromCPU(dLdOutput[0]);

                // Load and execute the kernel
                var gradientKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, double, double, ArrayView<double>, int>(GradientKernel);
                gradientKernel(new Index1D(numCols), dSoftmaxGpu.View, dPreviousOutputGpu.View, dCurrentdLdOutputGpu.View, dXGpu.View, dTempGpu.View, this.maskThreshold, this.temperature, dInputGpu.View, numCols);
                accelerator.Synchronize();

                dXGpu.CopyToCPU(dXParallel[iteration]);
                dTempGpu.CopyToCPU(dTempParallel[iteration]);
            }

            dSoftmaxGpu.Dispose();
            dPreviousOutputGpu.Dispose();
            dCurrentdLdOutputGpu.Dispose();
            dXGpu.Dispose();
            dTempGpu.Dispose();
            dInputGpu.Dispose();

            dX[0] = dXParallel.ElementwiseSumRows();
            dTemp[0] = dTempParallel.ElementwiseSumRows();

            return new BackwardResultBuilder()
                .AddInputGradient(dX)
                .AddInputGradient(dTemp)
                .Build();
        }
    }
}
