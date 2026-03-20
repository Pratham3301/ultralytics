# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from pathlib import Path
import unittest.mock as mock
import torch
import numpy as np

# Add local ultralytics to sys.path
local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if local_path not in sys.path:
    sys.path.insert(0, local_path)

import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results

def custom_fitness(metrics):
    """Custom fitness returns only mAP50."""
    return metrics.get('metrics/mAP50(B)', 0.0)

def test_custom_fitness_callable():
    """Test custom fitness function (callable) in YOLO training."""
    model = YOLO("yolov8n.yaml")
    print("Testing callable fitness_func...")
    model.train(data="coco8.yaml", imgsz=32, epochs=1, save=False, fitness_func=custom_fitness)
    trainer = model.trainer
    expected = custom_fitness(trainer.metrics)
    assert abs(expected - trainer.fitness) < 1e-4
    print("Callable fitness_func test PASSED!")

def test_custom_fitness_string():
    """Test custom fitness function (string) in YOLO training."""
    model = YOLO("yolov8n.yaml")
    print("Testing string fitness_func...")
    metric_name = 'metrics/mAP50(B)'
    model.train(data="coco8.yaml", imgsz=32, epochs=1, save=False, fitness_func=metric_name)
    trainer = model.trainer
    expected = trainer.metrics.get(metric_name, 0.0)
    assert abs(expected - trainer.fitness) < 1e-4
    print("String fitness_func test PASSED!")

def test_polars_fallback():
    """Test Polars fallback logic in plotting and results export."""
    print("Testing Polars fallbacks...")
    # Mock polars as missing
    with mock.patch.dict('sys.modules', {'polars': None}):
        # 1. Test plot_labels fallback
        from ultralytics.utils.plotting import plot_labels
        boxes = np.random.rand(10, 4)
        cls = np.random.randint(0, 2, 10)
        # Should not crash
        plot_labels(boxes, cls, names={0: 'a', 1: 'b'}, save_dir=Path('.'))
        print("plot_labels fallback PASSED!")

        # 2. Test to_csv fallback
        res = Results(orig_img=np.zeros((640, 640, 3), dtype=np.uint8), 
                      path='test.jpg', 
                      names={0: 'person'}, 
                      boxes=torch.tensor([[10, 10, 50, 50, 0.9, 0]]))
        csv_out = res.to_csv()
        assert isinstance(csv_out, str)
        assert 'name' in csv_out or 'class' in csv_out
        print("to_csv fallback PASSED!")

if __name__ == "__main__":
    try:
        test_custom_fitness_callable()
        test_custom_fitness_string()
        test_polars_fallback()
        print("\nAll refined tests PASSED!")
    except Exception as e:
        print(f"\nTests FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
