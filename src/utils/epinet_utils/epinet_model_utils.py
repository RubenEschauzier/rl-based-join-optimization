def inspect_ensemble_params(ensemble_gnn, ensemble_plan_cost):
    total_params = 0
    all_param_entries = []

    # Helper function to process a list of models
    def process_model(model, model_name):
        nonlocal total_params
        for name, param in model.named_parameters():
            count = param.numel()
            total_params += count

            # Create a descriptive name: "ensemble_gnn.conv1.weight"
            full_name = f"{model_name}.{name}"
            all_param_entries.append((full_name, count))

    # 1. Process both parts of your ensemble
    process_model(ensemble_gnn, "ensemble_gnn")
    process_model(ensemble_plan_cost, "ensemble_plan_cost")

    # 2. Sort by parameter count (Descending)
    all_param_entries.sort(key=lambda x: x[1], reverse=True)

    # 3. Print Top 5
    print("-" * 60)
    print(f"{'Parameter name':<40} | {'count':<10}")
    print("-" * 60)
    for name, count in all_param_entries[:5]:
        print(f"{name:<40} | {count:<10}")

    # 4. Print Total
    print("-" * 60)
    print(f"Single prior gnn parameters: {total_params:,}")
    print("-" * 60)
    return total_params