//using ParallelReverseAutoDiff.PRAD;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Linq.Expressions;
//using System.Text;
//using System.Threading.Tasks;

//namespace ParallelReverseAutoDiff.Test.AutoPrad
//{
//    public class DexBuilder
//    {
//        private readonly Dex dex;
//        private ComputationNode currentNode;

//        internal DexBuilder(Dex dex, ComputationNode startNode)
//        {
//            this.dex = dex;
//            this.currentNode = startNode;
//            dex.SetCurrentGraph(startNode);
//        }

//        public DexBuilder Then(Expression<Func<DexExp, DexExp>> operation)
//        {
//            // Replace the parameter in the expression with our current node
//            var parameter = operation.Parameters[0];
//            var visitor = new ParameterReplacementVisitor(parameter, new DexExp(currentNode));
//            var newBody = visitor.Visit(operation.Body);

//            var node = dex.ParseExpression(newBody);
//            currentNode = node;
//            dex.SetCurrentGraph(node);

//            return this;
//        }

//        public DexBuilder DoOp(string operation, Tensor t)
//        {
//            var node = new ComputationNode(operation, currentNode, new ComputationNode(t));
//            currentNode = node;
//            dex.SetCurrentGraph(node);
//            return this;
//        }

//        public DexExp Build() => new DexExp(currentNode);

//        public Tensor Forward() => dex.Forward();
//    }
//}
