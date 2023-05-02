namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    public class TimeStep
    {
        public List<OperationInfo> StartOperations { get; set; }

        public List<Layer> Layers { get; set; }

        public List<OperationInfo> EndOperations { get; set; }
    }
}
