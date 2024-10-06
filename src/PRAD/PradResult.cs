//------------------------------------------------------------------------------
// <copyright file="PradResult.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Reflection;
    using System.Runtime.CompilerServices;
    using System.Threading.Tasks;
    using ParallelReverseAutoDiff.RMAD;
    using static ParallelReverseAutoDiff.PRAD.PradOp;

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
        /// Create multiple branches in the computation graph.
        /// </summary>
        /// <param name="n">The number of branches.</param>
        /// <returns>The branch stack.</returns>
        public BranchStack BranchStack(int n)
        {
            return this.PradOp.BranchStack(n);
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<PradResult> operation)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<PradResult>>(operation);
            return instanceOperation();
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="other">The other tensor.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<Tensor, PradResult> operation, Tensor other)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<Tensor, PradResult>>(operation);
            return instanceOperation(other);
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="other">The other tensor.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<object, PradResult> operation, float other)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<object, PradResult>>(operation);
            return instanceOperation(other);
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="other">The other tensor.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<object, PradResult> operation, double other)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<object, PradResult>>(operation);
            return instanceOperation(other);
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="other">The other tensor.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<object, PradResult> operation, Tensor other)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<object, PradResult>>(operation);
            return instanceOperation(other);
        }

        /// <summary>
        /// Applies the following operation.
        /// Allows for this: x.Then(PradOp.SquareRoot).Then(PradOp.Add, y);.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="other">The other tensor.</param>
        /// <param name="additional">The additional tensor.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<Tensor, Tensor, PradResult> operation, Tensor other, Tensor additional)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<Tensor, Tensor, PradResult>>(operation);
            return instanceOperation(other, additional);
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
        /// <param name="others">The other tensors.</param>
        /// <param name="axis">The axis to concatenate on.</param>
        /// <param name="sliceSizes">The slice sizes to extract.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<Tensor[], int, int[], PradResult> operation, Tensor[] others, int axis, params int[] sliceSizes)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<Tensor[], int, int[], PradResult>>(operation);
            return instanceOperation(others, axis, sliceSizes);
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
        /// <param name="skip">The skip.</param>
        /// <param name="restart">The restart.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<int, int, PradResult> operation, int skip, int restart)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<int, int, PradResult>>(operation);
            return instanceOperation(skip, restart);
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

        /// <summary>
        /// Applies a custom operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="forward">The forward function.</param>
        /// <param name="backward">The backward function.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<Func<Tensor, Tensor>, Func<Tensor, Tensor, Tensor, Tensor[]>, PradResult> operation, Func<Tensor, Tensor> forward, Func<Tensor, Tensor, Tensor, Tensor[]> backward)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<Func<Tensor, Tensor>, Func<Tensor, Tensor, Tensor, Tensor[]>, PradResult>>(operation);
            return instanceOperation(forward, backward);
        }

        /// <summary>
        /// Applies a custom TensorOp operation.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <param name="op">The custom operation.</param>
        /// <param name="otherTensors">The other tensors involved.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then(Func<TensorOp, Tensor[], PradResult> operation, TensorOp op, params Tensor[] otherTensors)
        {
            var instanceOperation = this.PradOp.GetOperation<Func<TensorOp, Tensor[], PradResult>>(operation);
            return instanceOperation(op, otherTensors);
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <returns>A new PradResult after applying the operation.</returns>
        public PradResult Then(Func<PradResult, PradResult> operation)
        {
            return operation(this);
        }

        /// <summary>
        /// Applies the specified operations to this PradResult.
        /// </summary>
        /// <param name="operation">The operation to apply.</param>
        /// <returns>New PradResult instances after applying the operation.</returns>
        public PradResult[] Then(Func<PradResult, PradResult[]> operation)
        {
            return operation(this);
        }

        /// <summary>
        /// Executes multiple operations in parallel on the current PradResult.
        /// </summary>
        /// <param name="operations">An array of functions, each taking a PradResult and returning a PradResult.</param>
        /// <returns>An array of PradResults, one for each parallel operation.</returns>
        /// <exception cref="ArgumentException">Thrown when no operations are provided.</exception>
        public PradResult[] ThenParallel(params Func<PradResult, PradResult>[] operations)
        {
            if (operations == null || operations.Length == 0)
            {
                throw new ArgumentException("At least one operation must be provided.", nameof(operations));
            }

            var pradOps = new PradOp[operations.Length];
            pradOps[0] = this.PradOp; // Use the current PradOp for the first operation

            // Create branched PradOps for subsequent operations
            for (int i = 1; i < operations.Length; i++)
            {
                pradOps[i] = new PradOp(this.PradOp.GetCurrentTensor());
            }

            var results = new PradResult[operations.Length];

            // Use Parallel.For to execute operations in parallel
            Parallel.For(0, operations.Length, i =>
            {
                var branchedResult = new PradResult(pradOps[i], this.ResultTensor, this.Gradients);
                results[i] = operations[i](branchedResult);
            });

            // Record split branches
            for (int i = 1; i < operations.Length; i++)
            {
                pradOps[i].RecordSplitBranch(pradOps[0]);
            }

            return results;
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <typeparam name="TOperation">The operation to apply.</typeparam>
        /// <typeparam name="TParamType">The type of the first parameter to the forward function.</typeparam>
        /// <typeparam name="TReturnType">The return type of the forward function.</typeparam>
        /// <param name="opFunc">The op.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then<TOperation, TParamType, TReturnType>(Func<PradOperationBase<TOperation, TParamType, TReturnType>> opFunc)
            where TOperation : PradOperationBase<TOperation, TParamType, TReturnType>
        {
            var operation = opFunc.Invoke();
            var opType = operation.GetType();
            IOperation op = operation;
            var forwardMethod = opType.GetMethod("Forward");

            return this.InnerThenGeneric(new object?[] { this.PradOp.GetCurrentTensor() }, typeof(TReturnType), forwardMethod, op, (v) => new BackwardResult[] { op.Backward(v.ToMatrix()) });
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <typeparam name="TOperation">The operation to apply.</typeparam>
        /// <typeparam name="TParam1Type">The type of the first parameter to the Forward function.</typeparam>
        /// <typeparam name="TParam2Type">The type of the second parameter to the Forward function.</typeparam>
        /// <typeparam name="TReturnType">The return type of the Forward function.</typeparam>
        /// <param name="opFunc">The operation.</param>
        /// <param name="param2">The second param.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then<TOperation, TParam1Type, TParam2Type, TReturnType>(Func<PradOperationBase<TOperation, TParam1Type, TParam2Type, TReturnType>> opFunc, TParam2Type param2)
            where TOperation : PradOperationBase<TOperation, TParam1Type, TParam2Type, TReturnType>
        {
            var operation = opFunc.Invoke();
            var opType = operation.GetType();
            IOperation op = operation;
            var forwardMethod = opType.GetMethod("Forward");

            return this.InnerThenGeneric(new object?[] { this.PradOp.GetCurrentTensor(), param2 }, typeof(TReturnType), forwardMethod, op, (v) => new BackwardResult[] { op.Backward(v.ToMatrix()) });
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <typeparam name="TOperation">The operation to apply.</typeparam>
        /// <typeparam name="TParamType">The type of the first parameter to the Forward function.</typeparam>
        /// <typeparam name="TReturnType">The return type of the Forward function.</typeparam>
        /// <param name="operation">The operation.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then<TOperation, TParamType, TReturnType>(PradOperationBase<TOperation, TParamType, TReturnType> operation)
            where TOperation : PradOperationBase<TOperation, TParamType, TReturnType>
        {
            var opType = operation.GetType();
            IOperation op = operation;
            var forwardMethod = opType.GetMethod("Forward");

            return this.InnerThenGeneric(new object?[] { this.PradOp.GetCurrentTensor() }, typeof(TReturnType), forwardMethod, op, (v) => new BackwardResult[] { op.Backward(v.ToMatrix()) });
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <typeparam name="TOperation">The operation to apply.</typeparam>
        /// <typeparam name="TParam1Type">The type of the first parameter to the Forward function.</typeparam>
        /// <typeparam name="TParam2Type">The type of the second parameter to the Forward function.</typeparam>
        /// <typeparam name="TReturnType">The return type of the Forward function.</typeparam>
        /// <param name="operation">The operation.</param>
        /// <param name="param2">The second param.</param>
        /// <returns>A PradResult.</returns>
        public PradResult Then<TOperation, TParam1Type, TParam2Type, TReturnType>(PradOperationBase<TOperation, TParam1Type, TParam2Type, TReturnType> operation, TParam2Type param2)
            where TOperation : PradOperationBase<TOperation, TParam1Type, TParam2Type, TReturnType>
        {
            var opType = operation.GetType();
            IOperation op = operation;
            var forwardMethod = opType.GetMethod("Forward");

            return this.InnerThenGeneric(new object?[] { this.PradOp.GetCurrentTensor(), param2 }, typeof(TReturnType), forwardMethod, op, (v) => new BackwardResult[] { op.Backward(v.ToMatrix()) });
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <typeparam name="TOperation">The operation to apply.</typeparam>
        /// <typeparam name="TParamType">The type of the first parameter to the Forward function.</typeparam>
        /// <typeparam name="TReturnType">The return type of the Forward function.</typeparam>
        /// <param name="operation">The operation.</param>
        /// <param name="param">The param type.</param>
        /// <returns>A PradResult.</returns>
        internal PradResult Then<TOperation, TParamType, TReturnType>(PradDeepOperationBase<TOperation, TParamType, TReturnType> operation, TParamType param)
            where TOperation : PradDeepOperationBase<TOperation, TParamType, TReturnType>
        {
            var opType = operation.GetType();
            IDeepOperation op = operation;
            var forwardMethod = opType.GetMethod("Forward");

            return this.InnerThenGeneric(new object?[] { param }, typeof(TReturnType), forwardMethod, op, (v) => new BackwardResult[] { op.Backward(v.ToDeepMatrix()) });
        }

        /// <summary>
        /// Applies the specified operation to this PradResult.
        /// </summary>
        /// <typeparam name="TOperation">The operation to apply.</typeparam>
        /// <typeparam name="TParamType">The type of the first parameter to the Forward function.</typeparam>
        /// <typeparam name="TReturnType">The return type of the Forward function.</typeparam>
        /// <param name="operation">The operation.</param>
        /// <param name="param">The param type.</param>
        /// <returns>A PradResult.</returns>
        internal PradResult Then<TOperation, TParamType, TReturnType>(PradBatchOperationBase<TOperation, TParamType, TReturnType> operation, TParamType param)
            where TOperation : PradBatchOperationBase<TOperation, TParamType, TReturnType>
        {
            var opType = operation.GetType();
            IBatchOperation op = operation;
            var forwardMethod = opType.GetMethod("Forward");

            return this.InnerThenGeneric(new object?[] { param }, typeof(TReturnType), forwardMethod, op, (v) => op.Backward(v.ToDeepMatrix()));
        }

        private PradResult InnerThenGeneric(object?[] parameters, Type returnType, MethodInfo forwardMethod, IOperationBase op, Func<Tensor, BackwardResult[]> backwardFunc)
        {
            Func<Tensor, Tensor> forward = (t) =>
            {
                var data = forwardMethod.Invoke(op, parameters);
                if (data is Tensor tensor)
                {
                    return tensor;
                }
                else if (data is Matrix matrix)
                {
                    return matrix.ToTensor();
                }

                throw new InvalidOperationException("Forward method must return a Matrix or a Tensor.");
            };

            Func<Tensor, Tensor, Tensor, Tensor[]> backward = (t, u, v) =>
            {
                var backwardResults = backwardFunc.Invoke(v);
                if (backwardResults[0].Item1 is Tensor tensor)
                {
                    return new Tensor[] { tensor };
                }

                if (backwardResults[0].Item1 is Matrix matrix)
                {
                    return new Tensor[] { matrix.ToTensor() };
                }

                throw new InvalidOperationException("Backward method must return a Matrix or a Tensor.");
            };

            return this.PradOp.CustomOperation(forward, backward);
        }
    }
}
