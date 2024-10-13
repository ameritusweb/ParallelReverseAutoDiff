//------------------------------------------------------------------------------
// <copyright file="PradOpBranchTracker.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Tracks branches of a computation graph.
    /// </summary>
    public class PradOpBranchTracker
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradOpBranchTracker"/> class.
        /// </summary>
        public PradOpBranchTracker()
        {
        }

        /// <summary>
        /// Gets the branch tracker ID.
        /// </summary>
        public Guid Id { get; private set; } = Guid.NewGuid();

        /// <summary>
        /// Gets visited branches.
        /// </summary>
        public List<PradOp> VisitedBranches { get; private set; } = new List<PradOp>();

        /// <summary>
        /// Runs the branches for a specific branch.
        /// </summary>
        /// <param name="branch">The branch to run for.</param>
        public void RunBranchesFor(PradOp branch)
        {
            while (branch.UpstreamGradient == null && this.VisitedBranches.Any(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null))
            {
                var visited = this.VisitedBranches.FirstOrDefault(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null);
                visited.Back();
            }
        }
    }
}
