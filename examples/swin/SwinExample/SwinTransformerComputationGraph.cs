using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample
{
    public class SwinTransformerComputationGraph : ComputationGraph
    {
        private readonly SwinTransformerNetwork network;
        private readonly SwinTransformerModel model;

        public SwinTransformerComputationGraph(SwinTransformerNetwork network)
            : base(network)
        {
            this.network = network;
            this.model = network.Model;
        }

        public async Task Initialize()
        {
            // Basic intermediates and weights
            this.AddIntermediate("Input", _ => this.network.Parameters.InputSequence[0])
                .AddIntermediate("Output", _ => this.network.Output)
                .AddWeight("patch_embed_proj", _ => model.GetWeight("patch_embedding", "", "patch_embed_proj"))
                .AddGradient("d_patch_embed_proj", _ => model.GetGradient("patch_embedding", "", "patch_embed_proj"))
                .AddWeight("patch_embed_norm_weight", _ => model.GetWeight("patch_embedding", "", "patch_embed_norm_weight"))
                .AddGradient("d_patch_embed_norm_weight", _ => model.GetGradient("patch_embedding", "", "patch_embed_norm_weight"))
                .AddWeight("patch_embed_norm_bias", _ => model.GetWeight("patch_embedding", "", "patch_embed_norm_bias"))
                .AddGradient("d_patch_embed_norm_bias", _ => model.GetGradient("patch_embedding", "", "patch_embed_norm_bias"));

            // Setup stage and block weights, gradients, and operation finders
            for (int stage = 0; stage < model.StageDepths.Length; stage++)
            {
                var stagePrefix = $"stage_{stage}";

                // Setup input finder for each stage
                this.AddOperationFinder($"{stagePrefix}_input", x => {
                    if (stage == 0)
                        return this[$"patch_embedding_bias_{x.Layer}_0"];
                    return this[$"patch_merging_{stage - 1}_proj_{x.Layer}_0"];
                });

                // Setup blocks within each stage
                for (int block = 0; block < model.StageDepths[stage]; block++)
                {
                    var blockPrefix = $"block_{block}";

                    // Add block weights and gradients
                    this.AddWeight($"{stagePrefix}_{blockPrefix}_qkv_weight", x => model.GetWeight(stagePrefix, blockPrefix, "qkv_weight"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_qkv_weight", x => model.GetGradient(stagePrefix, blockPrefix, "qkv_weight"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_proj_weight", x => model.GetWeight(stagePrefix, blockPrefix, "proj_weight"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_proj_weight", x => model.GetGradient(stagePrefix, blockPrefix, "proj_weight"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_proj_bias", x => model.GetWeight(stagePrefix, blockPrefix, "proj_bias"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_proj_bias", x => model.GetGradient(stagePrefix, blockPrefix, "proj_bias"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_norm1_weight", x => model.GetWeight(stagePrefix, blockPrefix, "norm1_weight"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_norm1_bias", x => model.GetGradient(stagePrefix, blockPrefix, "norm1_bias"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_norm2_weight", x => model.GetWeight(stagePrefix, blockPrefix, "norm2_weight"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_norm2_bias", x => model.GetGradient(stagePrefix, blockPrefix, "norm2_bias"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_mlp_fc1_weight", x => model.GetWeight(stagePrefix, blockPrefix, "mlp_fc1_weight"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_mlp_fc1_weight", x => model.GetGradient(stagePrefix, blockPrefix, "mlp_fc1_weight"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_mlp_fc1_bias", x => model.GetWeight(stagePrefix, blockPrefix, "mlp_fc1_bias"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_mlp_fc1_bias", x => model.GetGradient(stagePrefix, blockPrefix, "mlp_fc1_bias"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_mlp_fc2_weight", x => model.GetWeight(stagePrefix, blockPrefix, "mlp_fc2_weight"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_mlp_fc2_weight", x => model.GetGradient(stagePrefix, blockPrefix, "mlp_fc2_weight"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_mlp_fc2_bias", x => model.GetWeight(stagePrefix, blockPrefix, "mlp_fc2_bias"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_mlp_fc2_bias", x => model.GetGradient(stagePrefix, blockPrefix, "mlp_fc2_bias"))
                        .AddWeight($"{stagePrefix}_{blockPrefix}_relative_position_bias_table", x => model.GetWeight(stagePrefix, blockPrefix, "relative_position_bias_table"))
                        .AddGradient($"d_{stagePrefix}_{blockPrefix}_relative_position_bias_table", x => model.GetGradient(stagePrefix, blockPrefix, "relative_position_bias_table"));

                    // Add block operation finders
                    this.AddOperationFinder($"{stagePrefix}_{blockPrefix}_input", x => {
                        if (block == 0)
                            return this[$"{stagePrefix}_input_{x.Layer}_0"];
                        return this[$"{stagePrefix}_block_{block - 1}_output_{x.Layer}_0"];
                    });

                    this.AddOperationFinder($"{stagePrefix}_{blockPrefix}_attention_input", x =>
                        this[$"{stagePrefix}_{blockPrefix}_norm1_{x.Layer}_0"]);

                    this.AddOperationFinder($"{stagePrefix}_{blockPrefix}_mlp_input", x =>
                        this[$"{stagePrefix}_{blockPrefix}_norm2_{x.Layer}_0"]);

                    this.AddOperationFinder($"{stagePrefix}_{blockPrefix}_shortcut", x =>
                        this[$"{stagePrefix}_{blockPrefix}_input_{x.Layer}_0"]);

                    if (block % 2 == 1)  // Shifted window attention for odd blocks
                    {
                        this.AddOperationFinder($"{stagePrefix}_{blockPrefix}_shifted_input", x =>
                            this[$"{stagePrefix}_{blockPrefix}_cyclic_shift_{x.Layer}_0"]);
                    }

                    this.AddOperationFinder($"{stagePrefix}_{blockPrefix}_window_attention", x =>
                        this[$"{stagePrefix}_{blockPrefix}_attention_{x.Layer}_0"]);
                }

                this.AddOperationFinder($"{stagePrefix}_output", x =>
                    this[$"{stagePrefix}_block_{model.StageDepths[stage] - 1}_output_{x.Layer}_0"]);
            }

            // Setup patch merging layers
            for (int i = 0; i < model.StageDepths.Length - 1; i++)
            {
                var mergePrefix = $"merge_{i}";

                this.AddWeight($"{mergePrefix}_proj_weight", x => model.GetWeight(mergePrefix, "", "proj_weight"))
                    .AddGradient($"d_{mergePrefix}_proj_weight", x => model.GetGradient(mergePrefix, "", "proj_weight"))
                    .AddWeight($"{mergePrefix}_norm_weight", x => model.GetWeight(mergePrefix, "", "norm_weight"))
                    .AddGradient($"d_{mergePrefix}_norm_weight", x => model.GetGradient(mergePrefix, "", "norm_weight"))
                    .AddWeight($"{mergePrefix}_norm_bias", x => model.GetWeight(mergePrefix, "", "norm_bias"))
                    .AddGradient($"d_{mergePrefix}_norm_bias", x => model.GetGradient(mergePrefix, "", "norm_bias"));

                this.AddOperationFinder($"{mergePrefix}_input", x =>
                    this[$"stage_{i}_output_{x.Layer}_0"]);
            }

            // Setup classification layer
            this.AddWeight("classifier_weight", x => model.GetWeight("classification", "", "classifier_weight"))
                .AddGradient("d_classifier_weight", x => model.GetGradient("classification", "", "classifier_weight"))
                .AddWeight("classifier_bias", x => model.GetWeight("classification", "", "classifier_bias"))
                .AddGradient("d_classifier_bias", x => model.GetGradient("classification", "", "classifier_bias"))
                .AddWeight("final_norm_weight", x => model.GetWeight("classification", "", "final_norm_weight"))
                .AddGradient("d_final_norm_weight", x => model.GetGradient("classification", "", "final_norm_weight"))
                .AddWeight("final_norm_bias", x => model.GetWeight("classification", "", "final_norm_bias"))
                .AddGradient("d_final_norm_bias", x => model.GetGradient("classification", "", "final_norm_bias"));

            // Add classification head finders
            this.AddOperationFinder("final_norm_input", x =>
                this[$"stage_{model.StageDepths.Length - 1}_output_{x.Layer}_0"]);

            this.AddOperationFinder("classification_input", x =>
                this[$"final_norm_{x.Layer}_0"]);

            // Read and construct from architecture JSON
            string json = File.ReadAllText("Architecture/swin-transformer.json");
            var jsonArchitecture = JsonConvert.DeserializeObject<FourLayersJsonArchitecture>(json)
                ?? throw new InvalidOperationException("Failed to deserialize architecture JSON");

            this.ConstructFromArchitecture(jsonArchitecture, 1, 2, 2, 6, 2, 0);

            // Setup backward dependency counts
            IOperationBase? backwardStartOperation = this["logits_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(
                Guid.NewGuid().ToString(),
                backwardStartOperation,
                0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        protected override void DependenciesSetup(IOperationBase operation, LayerInfo layerInfo)
        {
            base.DependenciesSetup(operation, layerInfo);
        }

        protected override Type TypeRetrieved(string type)
        {
            var baseType = base.TypeRetrieved(type);
            var customType = Type.GetType($"SwinExample.RMAD.{type}");
            return customType ?? baseType;
        }
    }
}
