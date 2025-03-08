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
                try
                {
                    var visited = this.VisitedBranches.FirstOrDefault(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null);
                    visited.Back();
                }
                catch (Exception)
                {
                }
            }
        }

        /// <summary>
        /// Process this branch.
        /// </summary>
        /// <param name="branchToCheck">The branch to check.</param>
        /// <param name="branch">The branch to process.</param>
        /// <param name="linkedBranch">The linked step.</param>
        public void ProcessBranch(PradOp branchToCheck, PradOp branch, PradOp? linkedBranch)
        {
            if (branch.UpstreamGradient != null && !branch.IsStarted && !branch.IsFinished)
            {
                branch.Back();
            }

            if (branchToCheck.UpstreamGradient == null)
            {
                while (branchToCheck.UpstreamGradient == null && branch.LinkedBranches.Any(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null))
                {
                    try
                    {
                        var visited = branch.LinkedBranches.FirstOrDefault(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null);
                        visited.Back();
                    }
                    catch (Exception)
                    {
                        throw new Exception("Branch could not be processed for backpropagation.");
                    }
                }

                if (branchToCheck.UpstreamGradient == null && branch.LastResult != null)
                {
                    var branches = branch.LastResult.Branches.ToList();
                    List<PradOp> moreBranches = new List<PradOp>();
                    foreach (var bbb in branches)
                    {
                        moreBranches.AddRange(bbb.LinkedBranches);
                    }

                    branches.AddRange(moreBranches);
                    branches = branches.Distinct().ToList();

                    while (branchToCheck.UpstreamGradient == null && branches.Any(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null))
                    {
                        try
                        {
                            var bb = branches.FirstOrDefault(f => !f.IsStarted && !f.IsFinished && f.UpstreamGradient != null);
                            bb.Back();
                        }
                        catch (Exception)
                        {
                            throw new Exception("Branch could not be processed for backpropagation.");
                        }
                    }
                }
            }

            if (branchToCheck.UpstreamGradient == null && linkedBranch != null)
            {
                this.ProcessBranch(branchToCheck, linkedBranch, default);
            }
        }
    }
}
