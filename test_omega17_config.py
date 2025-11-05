#!/usr/bin/env python3
"""
Comprehensive test for Omega17 model config and registry.

This test validates that:
1. Config loads without "Transformers does not recognize" error
2. Correct config classes are instantiated
3. Model architecture is recognized in registry
4. All required components are present

Run with: python test_omega17_config.py
"""

import sys
import traceback
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_config_loading(model_path):
    """Test 1: Config loading with registry mappings."""
    print_section("TEST 1: Config Loading")
    
    try:
        from vllm.transformers_utils.config import get_config
        
        print(f"Loading config from: {model_path}")
        print("Using trust_remote_code=False (no custom code execution)")
        
        config = get_config(
            model=model_path,
            trust_remote_code=False,
        )
        
        print(f"✓ Config loaded successfully!")
        print(f"  - Config class: {type(config).__name__}")
        print(f"  - Model type: {config.model_type}")
        
        # Check text_config
        if hasattr(config, 'text_config'):
            print(f"  - Text config class: {type(config.text_config).__name__}")
            print(f"  - Text model type: {config.text_config.model_type}")
        
        # Check vision_config
        if hasattr(config, 'vision_config'):
            print(f"  - Vision config present: Yes")
            print(f"  - Vision model type: {config.vision_config.model_type}")
        
        return config, True
        
    except Exception as e:
        print(f"✗ FAILED to load config!")
        print(f"  Error: {e}")
        traceback.print_exc()
        return None, False


def test_config_registry():
    """Test 2: Validate registry mappings."""
    print_section("TEST 2: Config Registry Mappings")
    
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        
        omega_types = {
            'omega17_vl': 'Qwen3VLConfig',
            'omega17_vl_exp': 'Qwen3VLMoeConfig', 
            'omega17_vl_exp_text': 'Qwen3NextConfig',
        }
        
        all_passed = True
        for model_type, expected_class in omega_types.items():
            if model_type in _CONFIG_REGISTRY:
                actual_class = _CONFIG_REGISTRY[model_type]
                if actual_class == expected_class:
                    print(f"✓ {model_type} → {expected_class}")
                else:
                    print(f"✗ {model_type} → {actual_class} (expected {expected_class})")
                    all_passed = False
            else:
                print(f"✗ {model_type} NOT FOUND in registry")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ FAILED to check registry!")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_model_registry(config):
    """Test 3: Model architecture resolution."""
    print_section("TEST 3: Model Architecture Registry")
    
    if config is None:
        print("⊘ Skipped - config loading failed")
        return False
    
    try:
        from vllm.config import ModelConfig
        from vllm.model_executor.models.registry import ModelRegistry
        
        # Get architecture from config
        architectures = config.architectures
        print(f"Model architectures: {architectures}")
        
        # Create a minimal ModelConfig
        model_config = ModelConfig(
            model="test",
            task="auto",
            tokenizer="test",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",
            seed=0,
        )
        model_config.hf_config = config
        
        # Try to resolve model class
        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            architectures,
            model_config
        )
        
        print(f"✓ Model architecture resolved!")
        print(f"  - Architecture: {resolved_arch}")
        print(f"  - Model class: {model_cls.__name__}")
        print(f"  - Module: {model_cls.__module__}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED to resolve model architecture!")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_imports():
    """Test 4: Required imports are available."""
    print_section("TEST 4: Import Validation")
    
    imports_to_test = [
        ("vllm.transformers_utils.configs", "Qwen3VLConfig"),
        ("vllm.transformers_utils.configs", "Qwen3VLMoeConfig"),
        ("vllm.transformers_utils.configs", "Qwen3NextConfig"),
        ("vllm.model_executor.models.omega17_vl_exp", "Omega17VLExpForConditionalGeneration"),
    ]
    
    all_passed = True
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name} - {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("=" * 70)
    print("  OMEGA17 MODEL CONFIG & REGISTRY TEST SUITE")
    print("=" * 70)
    
    # Default model path (can be overridden via command line)
    model_path = "arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print(f"\nModel: {model_path}")
    print(f"Python: {sys.version}")
    
    # Run tests
    results = {}
    
    # Test 4: Imports (run first as it's a prerequisite)
    results['imports'] = test_imports()
    
    # Test 2: Registry (can run without model download)
    results['registry'] = test_config_registry()
    
    # Test 1: Config loading (may download model if remote)
    config, results['config'] = test_config_loading(model_path)
    
    # Test 3: Model registry (depends on config)
    results['model_registry'] = test_model_registry(config)
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"{icon} {test_name.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The Omega17 fix is working correctly.")
        print("\nYou can now run your vLLM server with:")
        print(f"  vllm serve {model_path} --dtype bfloat16 ...")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())