//------------------------------------------------------------------------------
// <copyright file="PradResult.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;

    /// <summary>
    /// The result of the computation.
    /// </summary>
    public class PradResult : PradResultBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradResult"/> class.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="result">The tensor result.</param>
        /// <param name="gradients">The gradient tensors.</param>
        public PradResult(PradOp operation, Tensor result, Tensor[] gradients)
        {
            this.PradOp = operation;
            this.ResultTensor = result;
            this.Gradients = gradients;
        }

        /// <summary>
        /// Gets the result of the computation.
        /// </summary>
        public Tensor Result
        {
            get
            {
                return new PradTensor(this.PradOp, this.ResultTensor);
            }
        }

        /// <summary>
        /// Gets or sets the result tensor.
        /// </summary>
        internal Tensor ResultTensor { get; set; }

        /// <summary>
        /// Multiples two results together.
        /// </summary>
        /// <param name="a">The first result.</param>
        /// <param name="b">The second result.</param>
        /// <returns>The output result.</returns>
        public static PradResult operator *(PradResult a, PradResult b)
        {
            if (a.PradOp.IsCurrentlyAssociated(a))
            {
                return a.PradOp.Mul(b.Result);
            }
            else
            {
                var branch = a.PradOp.BranchAfterTheFact(a);
                return branch.Mul(b.Result);
            }
        }

        /// <summary>
        /// Adds two results together.
        /// </summary>
        /// <param name="a">The first result.</param>
        /// <param name="b">The second result.</param>
        /// <returns>The output result.</returns>
        public static PradResult operator +(PradResult a, PradResult b)
        {
            if (a.PradOp.IsCurrentlyAssociated(a))
            {
                return a.PradOp.Add(b.Result);
            }
            else
            {
                var branch = a.PradOp.BranchAfterTheFact(a);
                return branch.Add(b.Result);
            }
        }

        /// <summary>
        /// Divides two results by one another.
        /// </summary>
        /// <param name="a">The first result.</param>
        /// <param name="b">The second result.</param>
        /// <returns>The output result.</returns>
        public static PradResult operator /(PradResult a, PradResult b)
        {
            if (a.PradOp.IsCurrentlyAssociated(a))
            {
                return a.PradOp.Div(b.Result);
            }
            else
            {
                var branch = a.PradOp.BranchAfterTheFact(a);
                return branch.Div(b.Result);
            }
        }

        /// <summary>
        /// Subtracts two results from one another.
        /// </summary>
        /// <param name="a">The first result.</param>
        /// <param name="b">The second result.</param>
        /// <returns>The output result.</returns>
        public static PradResult operator -(PradResult a, PradResult b)
        {
            if (a.PradOp.IsCurrentlyAssociated(a))
            {
                return a.PradOp.Sub(b.Result);
            }
            else
            {
                var branch = a.PradOp.BranchAfterTheFact(a);
                return branch.Sub(b.Result);
            }
        }

        /// <summary>
        /// Backpropagates the gradient.
        /// </summary>
        /// <param name="upstreamGradient">The upstream gradient.</param>
        /// <returns>The gradient.</returns>
        public Tensor Back(Tensor upstreamGradient)
        {
            return this.PradOp.Back(upstreamGradient);
        }

        /// <summary>
        /// Create a branch in the computation graph.
        /// </summary>
        /// <returns>A PradOp.</returns>
        public PradOp Branch()
        {
            return this.PradOp.Branch();
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="other">The other tensor, if needed.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Delegate operation, Tensor? other = null)
        {
            if (other == null)
            {
                var instanceOperation = this.PradOp.GetOperation<Func<PradResult>>(operation);
                return instanceOperation();
            }
            else
            {
                var instanceOperation = this.PradOp.GetOperation<Func<Tensor, PradResult>>(operation);
                return instanceOperation(other);
            }
        }

        /// <summary>
        /// Applies the following operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="others">The other tensors.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<Tensor[], int, PradResult> operation, Tensor[] others, int axis = 0)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<Tensor[], int, PradResult>>(operation);
            return instanceOperation(others, axis);
        }

        /// <summary>
        /// Applies the following operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="indices">The indices.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<Tensor, int, PradResult> operation, Tensor indices, int axis = 0)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<Tensor, int, PradResult>>(operation);
            return instanceOperation(indices, axis);
        }

        /// <summary>
        /// Applies the following operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<int, PradResult> operation, int axis = -1)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<int, PradResult>>(operation);
            return instanceOperation(axis);
        }

        /// <summary>
        /// Applies the following operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<double, double, PradResult> operation, double min, double max)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<double, double, PradResult>>(operation);
            return instanceOperation(min, max);
        }

        /// <summary>
        /// Applies the following operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="array">The array.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<int[], PradResult> operation, params int[] array)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<int[], PradResult>>(operation);
            return instanceOperation(array);
        }

        /// <summary>
        /// Applies the following operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="array">The array.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<string[], PradResult> operation, params string[] array)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<string[], PradResult>>(operation);
            return instanceOperation(array);
        }
    }
}
