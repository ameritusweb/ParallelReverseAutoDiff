namespace ParallelReverseAutoDiff.RMAD
{
    public class OperationInfo
    {
        public string Id { get; set; }

        public string Description { get; set; }

        public string Type { get; set; }

        public string[] Inputs { get; set; }

        public string SetResultTo { get; set; }

        public string[] GradientResultTo { get; set; }
    }
}
