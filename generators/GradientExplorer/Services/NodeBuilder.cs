using System.Collections.Generic;

namespace GradientExplorer.Model
{
    public class NodeBuilder
    {
        private readonly Node node;
        public NodeType NodeType { get; set; }

        public NodeBuilder(NodeType type)
        {
            node = new Node { NodeType = type };
        }

        public NodeBuilder WithLeftOperand(Node left)
        {
            node.Edges.Add(new Edge
            {
                Relationship = RelationshipType.Operand,
                TargetNode = left
            });
            return this;
        }

        public NodeBuilder WithRightOperand(Node right)
        {
            node.Edges.Add(new Edge
            {
                Relationship = RelationshipType.Operand,
                TargetNode = right
            });
            return this;
        }

        public NodeBuilder WithBaseOperand(Node left)
        {
            node.Edges.Add(new Edge
            {
                Relationship = RelationshipType.Operand,
                TargetNode = left
            });
            return this;
        }

        public NodeBuilder WithExponentOperand(Node right)
        {
            node.Edges.Add(new Edge
            {
                Relationship = RelationshipType.Operand,
                TargetNode = right
            });
            return this;
        }

        public NodeBuilder WithOperands(List<Node> targets)
        {
            foreach (var target in targets)
            {
                node.Edges.Add(new Edge
                {
                    Relationship = RelationshipType.Operand,
                    TargetNode = target
                });
            }
            return this;
        }

        public NodeBuilder WithFunction(Node func)
        {
            node.Edges.Add(new Edge
            {
                Relationship = RelationshipType.Function,
                TargetNode = func
            });
            return this;
        }

        public Node Build()
        {
            // Additional validation or finalization logic can go here
            return node;
        }

        public void Reset()
        {
            this.NodeType = default;
            this.node.Edges.Clear();
        }
    }
}
