using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Diagram
{
    using System;
    using System.Threading;

    public class DiagramUniqueIDGenerator
    {
        private static readonly Lazy<DiagramUniqueIDGenerator> LazyLoadedInstance =
            new Lazy<DiagramUniqueIDGenerator>(() => new DiagramUniqueIDGenerator(), true);

        private int currentIntID;
        private long currentLongID;
        private IDType currentIDType;

        private enum IDType
        {
            Int,
            Long,
            Guid
        }

        public static DiagramUniqueIDGenerator Instance => LazyLoadedInstance.Value;

        private DiagramUniqueIDGenerator()
        {
            currentIDType = IDType.Int;
        }

        public string GetNextID()
        {
            switch (currentIDType)
            {
                case IDType.Int:
                    int nextIntID = Interlocked.Increment(ref currentIntID);
                    if (nextIntID == int.MaxValue)
                    {
                        currentIDType = IDType.Long;
                        Interlocked.Exchange(ref currentIntID, 0);
                    }
                    return nextIntID.ToString();

                case IDType.Long:
                    long nextLongID = Interlocked.Increment(ref currentLongID);
                    if (nextLongID == long.MaxValue)
                    {
                        currentIDType = IDType.Guid;
                        Interlocked.Exchange(ref currentLongID, 0);
                    }
                    return nextLongID.ToString();

                case IDType.Guid:
                    return Guid.NewGuid().ToString();

                default:
                    throw new InvalidOperationException("Unknown ID type.");
            }
        }
    }
}
