using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public class UniqueTypeSet
    {
        private readonly Dictionary<Type, object> _uniqueTypeObjects = new Dictionary<Type, object>();

        public object this[Type type]
        {
            get => _uniqueTypeObjects.TryGetValue(type, out var value) ? value : null;
            set
            {
                if (value != null && value.GetType() != type)
                {
                    throw new ArgumentException("The type of the value must match the indexer type.");
                }

                if (value == null)
                {
                    _uniqueTypeObjects.Remove(type);
                }
                else
                {
                    _uniqueTypeObjects[type] = value;
                }
            }
        }

        public bool ContainsType(Type type)
        {
            return _uniqueTypeObjects.ContainsKey(type);
        }
    }
}
