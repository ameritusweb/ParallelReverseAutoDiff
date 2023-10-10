using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
    public class HandlesAttribute : Attribute
    {
        public Type EventType { get; }

        public HandlesAttribute(Type eventType)
        {
            EventType = eventType;
        }
    }

}
