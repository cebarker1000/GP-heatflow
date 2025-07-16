from analysis.uq_wrapper import (
        build_fpca_model, save_fpca_model,
        recast_training_data_to_fpca)

fpca_model = build_fpca_model(input_file="outputs/uq_batch_results.npz", min_components=5, variance_threshold=0.99)

save_fpca_model(fpca_model, "outputs/fpca_model.npz")

recast_data = recast_training_data_to_fpca(
        input_file="outputs/uq_batch_results.npz",
        fpca_model=fpca_model,
        output_file="outputs/training_data_fpca.npz")