using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class GravNetTrainer
    {
        public async Task Train()
        {
            GravNet net = new GravNet(10, 200, 3, 0.0002d, 4d);
            await net.Initialize();

            Random rand = new Random(1234);
            Matrix input = new Matrix(100, 200);
            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {

                    input[i, j] = 1d;
                }
            }
            
            net.Forward();
        }
    }
}
