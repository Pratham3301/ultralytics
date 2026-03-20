# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from pathlib import Path

import torch

# Add local ultralytics to sys.path
# This ensures we test the local changes, not the installed package
local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, local_path)

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

def custom_fitness(metrics):
    """Custom fitness returns only mAP50."""
    print(f"Custom fitness called with metrics: {list(metrics.keys())}")
    return metrics.get('metrics/mAP50(B)', 0.0)

def test_custom_fitness():
    """Test custom fitness function in YOLO training."""
    # Load a small model
    # Use 'yolov8n.yaml' as it's a standard model in Ultralytics
    model = YOLO("yolov8n.yaml")
    
    # Train with custom fitness
    # Note: imgsz=32 and epochs=1 for speed
    # We use a small dataset 'coco8.yaml'
    print("Starting training with custom fitness function...")
    model.train(data="coco8.yaml", imgsz=32, epochs=1, save=True, fitness_func=custom_fitness)
    
    # After trainer.train(), the trainer object is available at model.trainer
    trainer = model.trainer
    
    # The final fitness used by the trainer should match the custom_fitness output for the last epoch
    expected_fitness = custom_fitness(trainer.metrics)
    actual_fitness = trainer.fitness
    
    print(f"Expected fitness (mAP50): {expected_fitness}")
    print(f"Actual fitness: {actual_fitness}")
    
    assert abs(expected_fitness - actual_fitness) < 1e-4, f"Fitness mismatch: {expected_fitness} != {actual_fitness}"
    print("Test PASSED!")

if __name__ == "__main__":
    try:
        test_custom_fitness()
    except Exception as e:
        print(f"Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
