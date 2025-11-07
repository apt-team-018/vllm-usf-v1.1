#!/usr/bin/env python3
"""
Quick compatibility test for Omega17VLExp model configuration
without downloading the model from HuggingFace.
"""

import json
import sys
from typing import Dict, Any

# Test config from the model
TEST_CONFIG = {
    "architectures": ["Omega17VLExpForConditionalGeneration"],
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "image_token_id": 151655,
    "model_type": "omega17_vl_exp",
    "pad_token_id": 151643,
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "omega17_vl_exp_text",
        "moe_intermediate_size": 768,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        },
        "rope_theta": 5000000,
        "router_aux_loss_coef": 0.001,
        "use_cache": False,
        "vocab_size": 151936
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.1",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "omega17_vl_exp",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652
}


def print_header(text: str, char: str = "="):
    """Print a formatted section header"""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}")


def test_imports() -> tuple[bool, list[str]]:
    """Test if required modules can be imported"""
    print_header("TEST 1: Import Validation", "=")
    
    imports_to_test = [
        ("vllm.transformers_utils.config", ["get_config", "_CONFIG_REGISTRY"]),
        ("vllm.model_executor.models.omega17_vl_exp", ["Omega17VLExpForConditionalGeneration"]),
        ("vllm.transformers_utils.configs.omega17_vl_exp", ["Omega17VLExpConfig", "Omega17VLExpTextConfig", "Omega17VLExpVisionConfig"]),
    ]
    
    failed = []
    passed = 0
    
    for module_name, attrs in imports_to_test:
        try:
            module = __import__(module_name, fromlist=attrs)
            for attr in attrs:
                if not hasattr(module, attr):
                    print(f"✗ {module_name}.{attr} - Attribute not found")
                    failed.append(f"{module_name}.{attr}")
                else:
                    print(f"✓ {module_name}.{attr}")
                    passed += 1
        except Exception as e:
            for attr in attrs:
                print(f"✗ {module_name}.{attr} - {str(e)}")
                failed.append(f"{module_name}.{attr}")
    
    success = len(failed) == 0
    print(f"\n{'✓' if success else '✗'} Imports: {'PASS' if success else 'FAIL'} ({passed} passed, {len(failed)} failed)")
    return success, failed


def test_config_registry() -> bool:
    """Test if omega17_vl_exp is registered in vLLM config registry"""
    print_header("TEST 2: Config Registry", "=")
    
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        
        model_type = "omega17_vl_exp"
        
        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            print(f"✓ Model type '{model_type}' is registered")
            print(f"  Config class: {config_class}")
            return True
        else:
            print(f"✗ Model type '{model_type}' NOT FOUND in registry")
            print(f"\nAvailable model types:")
            for key in sorted(_CONFIG_REGISTRY.keys()):
                if 'omega' in key.lower() or 'qwen' in key.lower():
                    print(f"  - {key}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED to check registry: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_registry() -> bool:
    """Test if Omega17VLExp model is registered"""
    print_header("TEST 3: Model Architecture Registry", "=")
    
    try:
        from vllm.model_executor.models import _MODELS, ModelRegistry
        
        model_arch = "Omega17VLExpForConditionalGeneration"
        
        # Check if registered
        if model_arch in _MODELS:
            print(f"✓ Architecture '{model_arch}' is registered")
            return True
        else:
            print(f"✗ Architecture '{model_arch}' NOT FOUND in model registry")
            print(f"\nSearching for similar models:")
            for key in sorted(_MODELS.keys()):
                if 'omega' in key.lower() or 'qwen' in key.lower():
                    print(f"  - {key}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED to check model registry: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_parsing() -> bool:
    """Test if the config can be parsed by vLLM's config classes"""
    print_header("TEST 4: Config Parsing", "=")
    
    try:
        from vllm.transformers_utils.configs.omega17_vl_exp import (
            Omega17VLExpConfig,
            Omega17VLExpTextConfig,
            Omega17VLExpVisionConfig
        )
        
        # Test text config
        print("Testing text config...")
        text_config = Omega17VLExpTextConfig(**TEST_CONFIG["text_config"])
        print(f"✓ Text config parsed successfully")
        print(f"  - hidden_size: {text_config.hidden_size}")
        print(f"  - num_hidden_layers: {text_config.num_hidden_layers}")
        print(f"  - num_experts: {text_config.num_experts}")
        
        # Test vision config
        print("\nTesting vision config...")
        vision_config = Omega17VLExpVisionConfig(**TEST_CONFIG["vision_config"])
        print(f"✓ Vision config parsed successfully")
        print(f"  - hidden_size: {vision_config.hidden_size}")
        print(f"  - depth: {vision_config.depth}")
        
        # Test main config
        print("\nTesting main config...")
        main_config = Omega17VLExpConfig(**TEST_CONFIG)
        print(f"✓ Main config parsed successfully")
        print(f"  - model_type: {main_config.model_type}")
        print(f"  - architectures: {main_config.architectures}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED to parse config: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  OMEGA17 MODEL COMPATIBILITY TEST")
    print("=" * 70)
    print(f"\nModel: arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020")
    print(f"Python: {sys.version}")
    
    results = {}
    
    # Run tests
    results['imports'], failed_imports = test_imports()
    results['config_registry'] = test_config_registry()
    results['model_registry'] = test_model_registry()
    results['config_parsing'] = test_config_parsing()
    
    # Summary
    print_header("TEST SUMMARY", "=")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ Model is COMPATIBLE with vLLM!")
        print("\nYou can proceed with building the Docker image.")
        return 0
    else:
        print("\n✗ Model has compatibility issues.")
        print("\nRecommendations:")
        
        if not results['imports']:
            print("  1. Missing imports - check that all config files exist:")
            print("     - vllm/transformers_utils/configs/omega17_vl_exp.py")
            print("     - vllm/model_executor/models/omega17_vl_exp.py")
        
        if not results['config_registry']:
            print("  2. Config not registered - add to vllm/transformers_utils/configs/__init__.py")
        
        if not results['model_registry']:
            print("  3. Model not registered - add to vllm/model_executor/models/__init__.py")
        
        if not results['config_parsing']:
            print("  4. Config parsing failed - check config class definitions")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
