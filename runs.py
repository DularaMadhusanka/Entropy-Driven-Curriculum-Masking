import resnet_experiments

# Configuration dictionary mapping model names and datasets to functions
# Structure: RUNS[model_name][dataset_name] = training_function
RUNS = {
    'resnet18': {
        # --- NEW: Your Novelty Experiment ---
        'oxford': resnet_experiments.train_oxford,
    }
}