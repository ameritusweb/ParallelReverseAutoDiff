using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace ParallelReverseAutoDiff.Test.AutoPrad
{
    public class AutoPradTests
    {
        [Fact]
        public void Test1()
        {
            AutoPrad autoPrad = new AutoPrad();
            var initialTensor = Tensor.XavierUniform(new int[] { 2, 10 });
            var otherTensor = Tensor.XavierUniform(new int[] { 2, 10 });
            var exp = autoPrad.DoOp(AutoPradOp.Add, initialTensor, otherTensor);
            var exp2 = autoPrad.DoOp(AutoPradOp.Mul, exp, otherTensor);
            var exp3 = autoPrad.DoOp(AutoPradOp.Div, exp, exp2);
            var exp4 = autoPrad.DoOp(() => exp3 * exp);
            var resTensor = autoPrad.Forward();
            exp4.BackAccumulate();
            var grad = autoPrad.Gradients[initialTensor];
        }
    }
}
