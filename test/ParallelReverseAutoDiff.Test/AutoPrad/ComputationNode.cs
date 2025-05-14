using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.AutoPrad
{
    public class ComputationNode
    {
        public enum NodeType { Tensor, Operation }
        public NodeType Type { get; }
        public string Operation { get; private set; }
        public ComputationNode Left { get; private set; }
        public ComputationNode Right { get; private set; }
        public Tensor TensorValue { get; }
        public bool IsReversed { get; private set; }

        // Usage tracking
        public int UsageCount { get; private set; }
        public HashSet<ComputationNode> Parents { get; } = new HashSet<ComputationNode>();
        public PradOp ResultOp { get; set; }
        private BranchStack BranchStack { get; set; }
        private int CurrentBranchIndex { get; set; }

        // For Tensor nodes
        public ComputationNode(Tensor value)
        {
            Type = NodeType.Tensor;
            TensorValue = value;
            UsageCount = 0;
        }

        // For Operation nodes
        public ComputationNode(string op, ComputationNode left, ComputationNode right)
        {
            Type = NodeType.Operation;
            Operation = op;
            Left = left;
            Right = right;
            UsageCount = 0;

            // Register this node as a parent of its children
            if (left != null) left.Parents.Add(this);
            if (right != null) right.Parents.Add(this);
        }

        public void AnalyzeUsage()
        {
            // Reset usage counts
            var visited = new HashSet<ComputationNode>();
            CalculateUsageCounts(visited);
        }

        private void CalculateUsageCounts(HashSet<ComputationNode> visited)
        {
            if (!visited.Add(this)) return;

            // For operation nodes, analyze children first
            if (Type == NodeType.Operation)
            {
                Left?.CalculateUsageCounts(visited);
                Right?.CalculateUsageCounts(visited);
            }

            // Usage count is the number of parents plus any external uses
            UsageCount = Parents.Count;
        }

        public PradOp GetOrCreateBranch()
        {
            if (ResultOp == null)
                throw new InvalidOperationException("Node has not been executed yet");

            if (UsageCount <= 1)
                return ResultOp;

            // First usage of a multi-used node
            if (BranchStack == null && CurrentBranchIndex == 0)
            {
                if (UsageCount == 2)
                {
                    CurrentBranchIndex++;
                    return ResultOp.Branch();
                }
                else
                {
                    BranchStack = ResultOp.BranchStack(UsageCount - 1);
                    CurrentBranchIndex++;
                    return BranchStack.Pop();
                }
            }
            // Subsequent usages
            else if (BranchStack != null && CurrentBranchIndex < UsageCount - 1)
            {
                CurrentBranchIndex++;
                return BranchStack.Pop();
            }
            // Last usage
            else
            {
                return ResultOp;
            }
        }

        public void RearrangeForIndependence()
        {
            if (Type == NodeType.Tensor) return;

            // Analyze if right node needs independence (has multiple parents)
            bool rightNeedsIndependence = Right.UsageCount > 1;

            if (rightNeedsIndependence)
            {
                switch (Operation)
                {
                    case DexOp.Add:
                        // For commutative operations, just swap the nodes
                        SwapNodes();
                        break;
                    case DexOp.Sub:
                        Operation = "SubFrom";
                        SwapNodes();
                        IsReversed = true;
                        break;
                    case DexOp.Div:
                        Operation = "DivInto";
                        SwapNodes();
                        IsReversed = true;
                        break;
                    case DexOp.Mul:
                        // Commutative operation
                        SwapNodes();
                        break;
                }
            }

            // Recursively rearrange children
            Left?.RearrangeForIndependence();
            Right?.RearrangeForIndependence();
        }

        private void SwapNodes()
        {
            var temp = Left;
            Left = Right;
            Right = temp;

            // Update parent references
            Left.Parents.Remove(this);
            Right.Parents.Remove(this);
            Left.Parents.Add(this);
            Right.Parents.Add(this);
        }

        public (PradOp op, bool isIndependent) BuildPradOp()
        {
            if (Type == NodeType.Tensor)
            {
                ResultOp = new PradOp(TensorValue);
                return (ResultOp, true);
            }

            var (leftOp, leftIndependent) = Left.BuildPradOp();

            // Get appropriate branch if the left node is used multiple times
            if (Left.UsageCount > 1)
                leftOp = Left.GetOrCreateBranch();

            var (rightOp, rightIndependent) = Right.BuildPradOp();

            // Get appropriate branch if the right node is used multiple times
            if (Right.UsageCount > 1)
                rightOp = Right.GetOrCreateBranch();

            PradResult result;
            switch (Operation)
            {
                case DexOp.Add:
                    result = leftOp.Add(rightOp.CurrentTensor);
                    break;
                case "SubFrom":
                    result = leftOp.SubFrom(rightOp.CurrentTensor);
                    break;
                case DexOp.Sub:
                    result = leftOp.Sub(rightOp.CurrentTensor);
                    break;
                case "DivInto":
                    result = leftOp.DivInto(rightOp.CurrentTensor);
                    break;
                case DexOp.Div:
                    result = leftOp.Div(rightOp.CurrentTensor);
                    break;
                case DexOp.Mul:
                    result = leftOp.Mul(rightOp.CurrentTensor);
                    break;
                default:
                    throw new ArgumentException($"Unknown operation: {Operation}");
            }

            ResultOp = result.PradOp;
            return (ResultOp, IsReversed || rightIndependent);
        }
    }
}
