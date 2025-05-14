//using ParallelReverseAutoDiff.PRAD;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Linq.Expressions;
//using System.Text;
//using System.Threading.Tasks;

//namespace ParallelReverseAutoDiff.Test.AutoPrad
//{
//    public class Dex
//    {
//        public DexGrad Gradients { get; set; }

//        public DexBuilder Seed(Tensor t) {

//            return new DexBuilder(this, new ComputationNode());
//        }

//        public DexExp DoOp(string operation, Tensor t1, Tensor t2)
//        {
//            return new DexExp();
//        }

//        public DexExp DoOp(string operation, DexExp e1, DexExp e2)
//        {
//            return new DexExp();
//        }

//        public DexExp DoOp(string operation, DexExp e1, Tensor t2)
//        {
//            return new DexExp();
//        }

//        public DexExp DoOp(Expression<Func<DexExp>> exp)
//        {
//            return new DexExp();
//        }

//        public Tensor Forward()
//        {
//            return null;
//        }

//        private ComputationNode currentGraph;
//        private readonly HashSet<Tensor> trackedTensors = new HashSet<Tensor>();
//        public AutoGrad Gradients { get; } = new AutoGrad();

//        // Original DoOp methods
//        public DexExp DoOp(string operation, Tensor t1, Tensor t2)
//        {
//            trackedTensors.Add(t1);
//            trackedTensors.Add(t2);
//            var node = new ComputationNode(operation,
//                new ComputationNode(t1),
//                new ComputationNode(t2));
//            currentGraph = node;
//            return new DexExp(node);
//        }

//        public DexExp DoOp(Expression<Func<DexExp>> exp)
//        {
//            var node = ParseExpression(exp.Body);
//            currentGraph = node;
//            return new DexExp(node);
//        }

//        // New fluent methods
//        public DexBuilder Start(Tensor t)
//        {
//            trackedTensors.Add(t);
//            return new DexBuilder(this, new ComputationNode(t));
//        }

//        public DexBuilder Start(Expression<Func<DexExp>> exp)
//        {
//            var node = ParseExpression(exp.Body);
//            return new DexBuilder(this, node);
//        }

//        internal void SetCurrentGraph(ComputationNode node)
//        {
//            currentGraph = node;
//        }

//        public Tensor Forward()
//        {
//            currentGraph.AnalyzeUsage();
//            currentGraph.RearrangeForIndependence();
//            var (finalOp, isIndependent) = currentGraph.BuildPradOp();

//            if (!isIndependent)
//                throw new InvalidOperationException("Failed to create independent final operation");

//            var finalExp = new DexExp(currentGraph);
//            finalExp.ResultOp = finalOp;

//            return finalOp.CurrentTensor;
//        }

//        private ComputationNode ParseExpression(Expression exp)
//        {
//            return exp switch
//            {
//                BinaryExpression binary => ParseBinaryExpression(binary),
//                MethodCallExpression method => ParseMethodCall(method),
//                MemberExpression member => ParseMemberAccess(member),
//                ConstantExpression constant => ParseConstant(constant),
//                _ => throw new ArgumentException($"Unsupported expression type: {exp.GetType()}")
//            };
//        }

//        private ComputationNode ParseBinaryExpression(BinaryExpression binary)
//        {
//            var left = ParseExpression(binary.Left);
//            var right = ParseExpression(binary.Right);

//            var operation = binary.NodeType switch
//            {
//                ExpressionType.Add => AutoPradOp.Add,
//                ExpressionType.Subtract => AutoPradOp.Sub,
//                ExpressionType.Multiply => AutoPradOp.Mul,
//                ExpressionType.Divide => AutoPradOp.Div,
//                _ => throw new ArgumentException($"Unsupported binary operation: {binary.NodeType}")
//            };

//            return new ComputationNode(operation, left, right);
//        }

//        private ComputationNode ParseMethodCall(MethodCallExpression method)
//        {
//            // Handle method calls like Tanh, Sigmoid, etc.
//            var target = method.Object != null ? ParseExpression(method.Object) : null;
//            var args = method.Arguments.Select(ParseExpression).ToList();

//            return method.Method.Name switch
//            {
//                "Tanh" => new ComputationNode("Tanh", target, null),
//                "Sigmoid" => new ComputationNode("Sigmoid", target, null),
//                "MatMul" => new ComputationNode(AutoPradOp.MatMul, args[0], args[1]),
//                // Add other operations as needed
//                _ => throw new ArgumentException($"Unsupported method: {method.Method.Name}")
//            };
//        }

//        private ComputationNode ParseMemberAccess(MemberExpression member)
//        {
//            // This would handle accessing PradExp variables
//            if (member.Expression is ConstantExpression constant)
//            {
//                var closure = constant.Value;
//                var field = closure.GetType().GetField(member.Member.Name);
//                var value = field.GetValue(closure);
//                if (value is DexExp pradExp)
//                {
//                    return pradExp.Node;
//                }
//            }
//            throw new ArgumentException($"Unsupported member access: {member}");
//        }

//        private ComputationNode ParseConstant(ConstantExpression constant)
//        {
//            if (constant.Value is Tensor tensor)
//            {
//                return new ComputationNode(tensor);
//            }
//            throw new ArgumentException($"Unsupported constant: {constant.Value}");
//        }
//    }
//}
