//------------------------------------------------------------------------------
// <copyright file="BranchStack.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Creates a stack of branches.
    /// </summary>
    public class BranchStack
    {
        private readonly Stack<PradOp> branches;

        /// <summary>
        /// Initializes a new instance of the <see cref="BranchStack"/> class.
        /// </summary>
        /// <param name="branches">The branches.</param>
        public BranchStack(IEnumerable<PradOp> branches)
        {
            this.branches = new Stack<PradOp>(branches);
        }

        /// <summary>
        /// Gets the number of branches remaining in the stack.
        /// </summary>
        public int Count => this.branches.Count;

        /// <summary>
        /// Pops a branch from the stack and returns it.
        /// </summary>
        /// <returns>The next PradOp branch from the stack.</returns>
        public PradOp Pop()
        {
            if (this.branches.Count == 0)
            {
                throw new InvalidOperationException("No more branches available in the stack.");
            }

            return this.branches.Pop();
        }

        /// <summary>
        /// Cleanup excess branches.
        /// </summary>
        public void Cleanup()
        {
            foreach (var branch in this.branches)
            {
                branch.TakeBackBranch();
            }

            this.branches.Clear();
        }
    }
}
