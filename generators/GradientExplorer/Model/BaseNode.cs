using GradientExplorer.Diagram;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class BaseNode
    {

        public string Id { get; private set; }

        public BaseNode() {

            this.Id = DiagramUniqueIDGenerator.Instance.GetNextID();
        
        }
    }
}
