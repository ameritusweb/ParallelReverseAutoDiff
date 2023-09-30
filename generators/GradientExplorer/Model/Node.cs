using Microsoft.CodeAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class Node
    {

        public Node()
        {

        }

        public Node(object rawValue, Type rawType)
        {
            RawValue = rawValue;
            RawType = rawType;
        }

        public FunctionType FunctionType { get; set; }

        public SyntaxNode SyntaxNode { get; set; }

        public string Type { get; set; } // "Constant", "Variable", "Operation", etc.
        public string Value { get; set; } // Value for constants, variable name for variables, etc.
        public object RawValue { get; set; }
        public Type RawType { get; set; }
        public List<Edge> Edges { get; set; } = new List<Edge>(); // Edges to other nodes
    }
}
