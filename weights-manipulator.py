#!/usr/bin/env python3
"""
Weight manipulator
Simple model manager with weight editing capabilities
This is an experimental study
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Suppress specific transformers warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        AutoConfig, BitsAndBytesConfig
    )
    from huggingface_hub import list_repo_files
except ImportError:
    print("Error: Install dependencies with:")
    print("pip install transformers torch huggingface_hub accelerate bitsandbytes")
    exit(1)

@dataclass
class WeightModification:
    """Class to track weight modifications"""
    layer_name: str
    token: str
    original_value: float
    new_value: float
    modification_type: str
    factor: float

class AdvancedModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Weight modification tracking
        self.modifications: List[WeightModification] = []
        self.system_prompt = "You are a helpful AI assistant."
        self.original_system_prompt = ""
        
        # Generation settings - CONFIGURABILI
        self.generation_settings = {
            'max_new_tokens': 300,      # Aumentato da 150 a 300
            'max_prompt_length': 1024,  # Aumentato da 512 a 1024
            'temperature': 0.7,
            'do_sample': True,
            'repetition_penalty': 1.1
        }
        
        # Useful token categories (base versions, variants auto-generated)
        self.useful_tokens = {
            "refusal": [
                "sorry", "cannot", "can't", "apologize", "unable", 
                "inappropriate", "harmful", "dangerous", "won't", 
                "refuse", "decline", "forbidden", "prohibited"
            ],
            "helpful": [
                "sure", "happy", "help", "certainly", "absolutely", 
                "assist", "understand", "course", "pleasure", "glad"
            ],
            "reasoning": [
                "think", "first", "however", "therefore", "because", 
                "since", "thus", "then", "next", "finally", "conclusion", 
                "analysis", "consider", "evaluate"
            ],
            "uncertainty": [
                "maybe", "perhaps", "possibly", "might", "could", 
                "uncertain", "probably", "seems", "appears", "likely"
            ],
            "confidence": [
                "definitely", "certainly", "clearly", "obviously", 
                "undoubtedly", "confident", "know", "sure", "absolutely", 
                "guaranteed", "precise", "exact"
            ],
            "questions": [
                "what", "how", "why", "when", "where", "who", "which", 
                "can", "could", "would", "should", "do", "does", "did"
            ],
            "negation": [
                "not", "no", "never", "nothing", "nobody", "nowhere", 
                "neither", "none", "nor", "hardly", "scarcely"
            ],
            "emotions": [
                "happy", "sad", "angry", "excited", "worried", "calm", 
                "surprised", "confused", "proud", "ashamed", "grateful", "frustrated"
            ]
        }
        
        print(f"🤖 Device: {self.device}")
        print(f"📁 Models directory: {self.models_dir}")

    def scan_local_models(self) -> List[str]:
        """Scan models/ directory for available models"""
        local_models = []
        
        if self.models_dir.exists():
            print(f"Scanning {self.models_dir}...")
            for item in self.models_dir.iterdir():
                if item.is_dir():
                    # Check for required model files
                    config_file = item / "config.json"
                    model_files = list(item.glob("*.bin")) + list(item.glob("*.safetensors"))
                    
                    if config_file.exists() and model_files:
                        local_models.append(item.name)
                        print(f"  ✓ Found: {item.name}")
                    else:
                        print(f"  ⚠️ Incomplete: {item.name}")
        else:
            print("models/ directory not found")
        
        return local_models

    def list_popular_models(self) -> List[str]:
        """List of important models from Hugging Face"""
        return [
            # Conversational models
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium", 
            "microsoft/DialoGPT-large",
            
            # GPT classics
            "gpt2",
            "gpt2-medium",
            "gpt2-large", 
            "gpt2-xl",
            "distilgpt2",
            
            # LLaMA and variants
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            
            # Mistral
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            
            # Code-specific
            "Salesforce/codegen-350M-multi",
            "microsoft/CodeGPT-small-py",
            "WizardLM/WizardCoder-1B-V1.0",
            
            # EleutherAI
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B", 
            "EleutherAI/gpt-j-6b",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            
            # OPT (Meta)
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            
            # Instruction-tuned
            "stabilityai/stablelm-tuned-alpha-3b",
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            
            # Specialized
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "google/flan-t5-small",
            "google/flan-t5-base",
            
            # Newer/Alternative
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/phi-1_5",
            "microsoft/phi-2"
        ]

    def download_model(self, model_name: str) -> bool:
        """Download model from Hugging Face and save locally"""
        try:
            print(f"📥 Downloading {model_name}...")
            
            # Create local directory
            local_path = self.models_dir / model_name.replace("/", "--")
            local_path.mkdir(exist_ok=True)
            
            # Download tokenizer
            print("  🔤 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.save_pretrained(local_path)
            
            # Download model
            print("  🧠 Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model.save_pretrained(local_path)
            
            print(f"✅ Model saved to: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return False

    def load_model(self, model_path: str, use_quantization: bool = False) -> bool:
        """Load model from local path or download if needed"""
        try:
            print(f"🔄 Loading {model_path}...")
            
            # Check if it's a local path or model name
            if (self.models_dir / model_path).exists():
                # Load from local
                full_path = self.models_dir / model_path
                print(f"  📂 Loading from local: {full_path}")
            else:
                # Download first
                model_folder = model_path.replace("/", "--")
                if not (self.models_dir / model_folder).exists():
                    if not self.download_model(model_path):
                        return False
                full_path = self.models_dir / model_folder
            
            # Load tokenizer with robust pad_token handling
            self.tokenizer = AutoTokenizer.from_pretrained(full_path, trust_remote_code=True)
            
            # Fix attention mask warning by properly setting pad_token
            if self.tokenizer.pad_token is None:
                # Strategy 1: Try unk_token first
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                    self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                    print("✓ Using unk_token as pad_token")
                
                # Strategy 2: Add new token if possible
                elif hasattr(self.tokenizer, 'add_special_tokens'):
                    try:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        print("✓ Added new pad_token: [PAD]")
                    except:
                        # Strategy 3: Use eos_token as last resort
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        print("⚠️  Using eos_token as pad_token (may cause warnings)")
                else:
                    # Strategy 3: Use eos_token as last resort
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    print("⚠️  Using eos_token as pad_token (may cause warnings)")
            
            # Debug info
            pad_eos_same = (self.tokenizer.pad_token_id == self.tokenizer.eos_token_id)
            print(f"Pad token: '{self.tokenizer.pad_token}' (ID: {self.tokenizer.pad_token_id})")
            print(f"EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")
            print(f"Pad==EOS: {pad_eos_same}")
            
            # Load model with quantization if CUDA
            quantization_config = None
            if use_quantization and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                full_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Enable weight modifications
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad_(True)
            
            self.model_name = model_path
            self.original_system_prompt = self.system_prompt
            print(f"✅ Model loaded successfully!")
            print(f"📊 Parameters: ~{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
            print(f"Mode: Training (modifications enabled)")
            return True
            
        except Exception as e:
            print(f"❌ Loading failed: {str(e)}")
            return False

    def get_similar_tokens_by_embedding(self, target_token: str, similarity_threshold: float = 0.7, max_tokens: int = 20) -> List[str]:
        """Find semantically similar tokens using cosine similarity of embeddings"""
        if not self.model or not self.tokenizer:
            return [target_token]
        
        try:
            # Get target token embedding
            target_token_id = self.tokenizer.encode(target_token, add_special_tokens=False)
            if not target_token_id:
                return [target_token]
            target_token_id = target_token_id[0]
            
            # Get embedding layer
            embedding_layer = None
            for name, param in self.model.named_parameters():
                if any(pattern in name.lower() for pattern in ['embed', 'wte', 'word_embed']):
                    if len(param.shape) >= 2:
                        embedding_layer = param
                        break
            
            if embedding_layer is None:
                print("⚠️ No embedding layer found")
                return [target_token]
            
            # Get target embedding
            if target_token_id >= embedding_layer.shape[0]:
                return [target_token]
            
            target_embedding = embedding_layer[target_token_id].detach()
            
            # Calculate cosine similarities with all tokens
            similarities = []
            vocab_size = min(embedding_layer.shape[0], 10000)  # Limit for performance
            
            print(f"🔍 Analyzing {vocab_size} tokens for similarity to '{target_token}'...")
            
            for token_id in range(vocab_size):
                if token_id == target_token_id:
                    continue
                
                try:
                    token_embedding = embedding_layer[token_id].detach()
                    
                    # Cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        target_embedding.unsqueeze(0), 
                        token_embedding.unsqueeze(0)
                    ).item()
                    
                    if cos_sim > similarity_threshold:
                        # Decode token
                        try:
                            token_text = self.tokenizer.decode([token_id])
                            # Filter out special tokens and weird characters
                            if (len(token_text.strip()) > 0 and 
                                not token_text.startswith(('<', '[', '#')) and 
                                token_text.isascii() and 
                                any(c.isalpha() for c in token_text)):
                                similarities.append((token_text.strip(), cos_sim, token_id))
                        except:
                            continue
                except:
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_tokens = [target_token]  # Include original
            
            for token_text, sim_score, token_id in similarities[:max_tokens]:
                similar_tokens.append(token_text)
                print(f"  📊 {token_text:<15} (similarity: {sim_score:.3f})")
            
            print(f"✅ Found {len(similar_tokens)-1} similar tokens to '{target_token}'")
            return similar_tokens
            
        except Exception as e:
            print(f"⚠️ Error finding similar tokens: {e}")
            return [target_token]

    def get_token_variants(self, token: str, use_semantic_similarity: bool = False, similarity_threshold: float = 0.7) -> List[str]:
        """Generate token variants - either case variants or semantic similarity"""
        if use_semantic_similarity:
            return self.get_similar_tokens_by_embedding(token, similarity_threshold)
        else:
            # Original case-based variants
            variants = []
            
            # Original token
            variants.append(token)
            
            # Lowercase version
            lower_variant = token.lower()
            if lower_variant != token:
                variants.append(lower_variant)
            
            # Uppercase version
            upper_variant = token.upper()
            if upper_variant != token:
                variants.append(upper_variant)
            
            # Capitalized version
            capitalize_variant = token.capitalize()
            if capitalize_variant not in variants:
                variants.append(capitalize_variant)
            
            # For longer tokens, add title case
            if len(token) > 3:
                title_variant = token.title()
                if title_variant not in variants:
                    variants.append(title_variant)
            
            return variants

    def apply_to_all_variants(self, base_tokens: List[str], use_semantic: bool = False) -> List[str]:
        """Apply variant generation to a list of base tokens"""
        all_variants = []
        for token in base_tokens:
            variants = self.get_token_variants(token, use_semantic_similarity=use_semantic)
            all_variants.extend(variants)
        
        # Remove duplicates while maintaining order
        seen = set()
        unique_variants = []
        for variant in all_variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants

    def get_token_weights(self, token: str) -> Dict[str, float]:
        """Get weights for a specific token"""
        if not self.model or not self.tokenizer:
            return {}
        
        try:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                return {}
            token_id = token_ids[0]
        except:
            print(f"Token '{token}' not found in vocabulary")
            return {}
        
        weights = {}
        
        # Search in embedding and output layers with broader patterns
        for name, param in self.model.named_parameters():
            if any(pattern in name.lower() for pattern in 
                   ['embed', 'lm_head', 'wte', 'word_embed', 'token_embed']):
                if len(param.shape) >= 2 and token_id < param.shape[0]:
                    try:
                        weights[name] = float(param[token_id].mean().item())
                    except:
                        continue
        
        return weights

    def modify_token_weights(self, token: str, modification_type: str, factor: float, include_variants: bool = False, use_semantic_similarity: bool = False, similarity_threshold: float = 0.7):
        """Modify weights for a token and optionally its variants or similar tokens"""
        if not self.model or not self.tokenizer:
            print("No model loaded")
            return
        
        # Determine tokens to modify
        tokens_to_modify = [token]
        if include_variants:
            if use_semantic_similarity:
                print(f"🧠 Finding semantically similar tokens to '{token}'...")
                tokens_to_modify = self.get_similar_tokens_by_embedding(token, similarity_threshold)
                print(f"📊 Will modify {len(tokens_to_modify)} semantically similar tokens")
            else:
                tokens_to_modify = self.get_token_variants(token, use_semantic_similarity=False)
                print(f"📝 Modifying case variants: {tokens_to_modify}")
        
        total_modifications = 0
        
        for current_token in tokens_to_modify:
            try:
                # Try to tokenize
                token_ids = self.tokenizer.encode(current_token, add_special_tokens=False)
                if not token_ids:
                    print(f"Token '{current_token}' cannot be tokenized")
                    continue
                
                token_id = token_ids[0]  # Take first token ID
                
            except Exception as e:
                print(f"Error tokenizing '{current_token}': {e}")
                continue
            
            modifications_made = 0
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    # Search in embedding and output layers
                    if any(pattern in name.lower() for pattern in 
                           ['embed', 'lm_head', 'wte', 'word_embed', 'token_embed']):
                        
                        if len(param.shape) >= 2 and token_id < param.shape[0]:
                            # Save original value
                            original_value = float(param[token_id].mean().item())
                            
                            # Apply modification
                            if modification_type == "zero":
                                param[token_id] = 0.0
                                new_value = 0.0
                            elif modification_type == "reduce":
                                param[token_id] *= (1.0 - factor)
                                new_value = float(param[token_id].mean().item())
                            elif modification_type == "boost":
                                param[token_id] *= (1.0 + factor)
                                new_value = float(param[token_id].mean().item())
                            
                            # Record modification
                            mod = WeightModification(
                                layer_name=name,
                                token=current_token,
                                original_value=original_value,
                                new_value=new_value,
                                modification_type=modification_type,
                                factor=factor
                            )
                            self.modifications.append(mod)
                            modifications_made += 1
                            
                            # Force parameter to require gradients
                            param.requires_grad_(True)
            
            if modifications_made > 0:
                total_modifications += modifications_made
            else:
                print(f"⚠️ No layers modified for '{current_token}'")
        
        print(f"✅ Total modifications: {total_modifications} across {len(tokens_to_modify)} tokens")

    def set_system_prompt(self, prompt: str):
        """Set custom system prompt"""
        if not self.original_system_prompt:
            self.original_system_prompt = self.system_prompt
        
        self.system_prompt = prompt
        print(f"✅ System prompt updated ({len(prompt)} characters)")

    def generation_settings_menu(self):
        """Menu for configuring generation settings"""
        print("\n--- GENERATION SETTINGS ---")
        print(f"Current settings:")
        for key, value in self.generation_settings.items():
            print(f"  {key}: {value}")
        
        print("\nOptions:")
        print("1. Change response length (max_new_tokens)")
        print("2. Change prompt length limit (max_prompt_length)")
        print("3. Change temperature (creativity)")
        print("4. Change repetition penalty")
        print("5. Reset to defaults")
        print("6. Load presets")
        print("0. Back")
        
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            try:
                new_length = int(input(f"New response length (current: {self.generation_settings['max_new_tokens']}): "))
                if 10 <= new_length <= 4096:
                    self.generation_settings['max_new_tokens'] = new_length
                    print(f"✅ Response length set to {new_length}")
                else:
                    print("❌ Length must be between 10 and 4096")
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == "2":
            try:
                new_length = int(input(f"New prompt length limit (current: {self.generation_settings['max_prompt_length']}): "))
                if 256 <= new_length <= 8192:
                    self.generation_settings['max_prompt_length'] = new_length
                    print(f"✅ Prompt length limit set to {new_length}")
                else:
                    print("❌ Length must be between 256 and 8192")
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == "3":
            try:
                new_temp = float(input(f"New temperature (current: {self.generation_settings['temperature']}): "))
                if 0.1 <= new_temp <= 2.0:
                    self.generation_settings['temperature'] = new_temp
                    print(f"✅ Temperature set to {new_temp}")
                else:
                    print("❌ Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == "4":
            try:
                new_penalty = float(input(f"New repetition penalty (current: {self.generation_settings['repetition_penalty']}): "))
                if 1.0 <= new_penalty <= 2.0:
                    self.generation_settings['repetition_penalty'] = new_penalty
                    print(f"✅ Repetition penalty set to {new_penalty}")
                else:
                    print("❌ Penalty must be between 1.0 and 2.0")
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == "5":
            self.generation_settings = {
                'max_new_tokens': 300,
                'max_prompt_length': 1024,
                'temperature': 0.7,
                'do_sample': True,
                'repetition_penalty': 1.1
            }
            print("✅ Settings reset to defaults")
        
        elif choice == "6":
            self.show_generation_presets()
        
        elif choice == "0":
            return
        
        # Show menu again after operation
        if choice != "0":
            input("\nPress Enter to continue...")
            self.generation_settings_menu()

    def show_generation_presets(self):
        """Show and apply generation presets"""
        presets = {
            "conservative": {
                'max_new_tokens': 200,
                'temperature': 0.3,
                'repetition_penalty': 1.2,
                'description': "Short, focused, conservative responses"
            },
            "balanced": {
                'max_new_tokens': 300,
                'temperature': 0.7,
                'repetition_penalty': 1.1,
                'description': "Balanced creativity and coherence"
            },
            "creative": {
                'max_new_tokens': 500,
                'temperature': 0.9,
                'repetition_penalty': 1.05,
                'description': "Creative, longer responses"
            },
            "detailed": {
                'max_new_tokens': 800,
                'temperature': 0.6,
                'repetition_penalty': 1.15,
                'description': "Long, detailed responses"
            },
            "coding": {
                'max_new_tokens': 600,
                'temperature': 0.2,
                'repetition_penalty': 1.1,
                'description': "Precise responses for coding"
            }
        }
        
        print("Available presets:")
        for i, (name, config) in enumerate(presets.items(), 1):
            print(f"  {i}. {name.title()}: {config['description']}")
            print(f"     Length: {config['max_new_tokens']}, Temp: {config['temperature']}")
        
        try:
            idx = int(input("Select preset: ")) - 1
            if 0 <= idx < len(presets):
                preset_name = list(presets.keys())[idx]
                preset = presets[preset_name]
                
                # Apply preset settings
                self.generation_settings['max_new_tokens'] = preset['max_new_tokens']
                self.generation_settings['temperature'] = preset['temperature']
                self.generation_settings['repetition_penalty'] = preset['repetition_penalty']
                
                print(f"✅ Applied '{preset_name}' preset")
            else:
                print("❌ Invalid selection!")
        except ValueError:
            print("❌ Invalid input!")
        """Show table of applied modifications"""
        if not self.modifications:
            print("No modifications applied to model")
            return
        
        print("\n" + "="*80)
        print("APPLIED MODIFICATIONS OVERVIEW")
        print("="*80)
        print(f"{'Token':<15} {'Layer':<25} {'Type':<10} {'Original':<10} {'New':<10} {'Factor':<10}")
        print("-"*80)
        
        for mod in self.modifications:
            print(f"{mod.token:<15} {mod.layer_name[-25:]:<25} {mod.modification_type:<10} "
                  f"{mod.original_value:<10.4f} {mod.new_value:<10.4f} {mod.factor:<10.2f}")
        
        if self.system_prompt and self.system_prompt != self.original_system_prompt:
            print("-"*80)
            print("MODIFIED SYSTEM PROMPT:")
            print(f"Original: {self.original_system_prompt[:50]}...")
            print(f"Current:  {self.system_prompt[:50]}...")
        
        print("="*80)

    def generate_response(self, user_input: str, max_length: Optional[int] = None) -> str:
        """Generate response using loaded model with configurable length"""
        if not self.model or not self.tokenizer:
            return "❌ No model loaded"
        
        # Use provided max_length or default from settings
        if max_length is None:
            max_length = self.generation_settings['max_new_tokens']
        
        try:
            # Format prompt with system and user messages
            if "chat" in self.model_name.lower() or "instruct" in self.model_name.lower():
                # Chat format for instruction-tuned models
                prompt = f"System: {self.system_prompt}\nUser: {user_input}\nAssistant:"
            else:
                # Simple format for base models
                prompt = f"{self.system_prompt}\n\nHuman: {user_input}\nAI:"
            
            # Suppress attention mask warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*attention mask.*")
                
                # Tokenize with explicit attention mask handling
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,  # No padding to avoid issues
                    truncation=True,
                    max_length=self.generation_settings['max_prompt_length'],  # Usa setting configurabile
                    return_attention_mask=True
                )
                
                # If pad_token == eos_token, create manual attention mask
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    # Create manual attention mask (all 1s for valid tokens)
                    attention_mask = torch.ones_like(inputs['input_ids'])
                    inputs['attention_mask'] = attention_mask
                
                # Move to correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    # Prepare generation kwargs based on model capabilities
                    generation_kwargs = {
                        'input_ids': inputs['input_ids'],
                        'attention_mask': inputs['attention_mask'],
                        'max_new_tokens': max_length,
                        'num_return_sequences': 1,
                        'temperature': self.generation_settings['temperature'],
                        'do_sample': self.generation_settings['do_sample'],
                        'pad_token_id': self.tokenizer.eos_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'use_cache': True
                    }
                    
                    # Add optional parameters only if supported
                    try:
                        # Test if advanced parameters are supported
                        if 'repetition_penalty' in self.generation_settings:
                            generation_kwargs['repetition_penalty'] = self.generation_settings['repetition_penalty']
                        generation_kwargs['no_repeat_ngram_size'] = 2
                        outputs = self.model.generate(**generation_kwargs)
                    except Exception:
                        # Fallback without advanced parameters
                        generation_kwargs = {
                            'input_ids': inputs['input_ids'],
                            'attention_mask': inputs['attention_mask'],
                            'max_new_tokens': max_length,
                            'num_return_sequences': 1,
                            'temperature': self.generation_settings['temperature'],
                            'do_sample': self.generation_settings['do_sample'],
                            'pad_token_id': self.tokenizer.eos_token_id,
                            'eos_token_id': self.tokenizer.eos_token_id
                        }
                        outputs = self.model.generate(**generation_kwargs)
                
                # Decode only the generated part
                input_length = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Clean up common artifacts
                response = response.strip()
                if response.startswith(("Human:", "User:", "AI:", "Assistant:")):
                    response = response.split(":", 1)[1].strip()
                
                return response if response else "🤔 No response generated"
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}"

    def weight_modification_menu(self):
        """Interactive weight modification menu"""
        if not self.model:
            print("❌ No model loaded!")
            return
        
        print("\n--- WEIGHT MODIFICATION MENU ---")
        print("1. Modify single token")
        print("2. Modify token category")
        print("3. Show current modifications")
        print("4. Save modifications config")
        print("5. Load modifications config")
        print("0. Back to main menu")
        
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            self.handle_single_token_modification()
        elif choice == "2":
            self.handle_category_modification()
        elif choice == "3":
            self.show_modifications_table()
        elif choice == "4":
            self.save_modifications_config()
        elif choice == "5":
            self.load_modifications_config()
        elif choice == "0":
            return
        else:
            print("Invalid option!")
        
        # Show menu again after operation
        input("\nPress Enter to continue...")
        self.weight_modification_menu()

    def handle_single_token_modification(self):
        """Handle single or multiple token modification with semantic similarity option"""
        token_input = input("Enter token(s) to modify (single token or comma-separated list): ").strip()
        if not token_input:
            return
        
        # Parse input - handle both single token and comma-separated list
        if ',' in token_input:
            tokens = [token.strip() for token in token_input.split(',') if token.strip()]
            print(f"📝 Processing {len(tokens)} tokens: {tokens}")
        else:
            tokens = [token_input.strip()]
            print(f"📝 Processing single token: {token_input}")
        
        # Show current weights for first token as example
        if tokens:
            first_token = tokens[0]
            current_weights = self.get_token_weights(first_token)
            if current_weights:
                print(f"\nExample weights for '{first_token}':")
                for layer, weight in list(current_weights.items())[:3]:
                    print(f"  {layer}: {weight:.4f}")
                if len(current_weights) > 3:
                    print(f"  ... and {len(current_weights)-3} more layers")
        
        # Choose similarity type
        print(f"\nVariant/similarity options:")
        print("1. No variants (just the specified token(s))")
        print("2. Case variants for each token (sorry → Sorry, SORRY)")
        print("3. Semantic similarity for each token (sorry → apologize, regret, afraid...)")
        
        variant_choice = input("Choose option: ").strip()
        
        include_variants = False
        use_semantic = False
        similarity_threshold = 0.7
        
        if variant_choice == "2":
            include_variants = True
            use_semantic = False
            # Preview variants for first few tokens
            preview_tokens = tokens[:3]
            for token in preview_tokens:
                variants = self.get_token_variants(token, use_semantic_similarity=False)
                print(f"  '{token}' variants: {variants}")
            if len(tokens) > 3:
                print(f"  ... and variants for {len(tokens)-3} more tokens")
                
        elif variant_choice == "3":
            include_variants = True
            use_semantic = True
            print("🧠 Semantic similarity analysis (this may take a moment)...")
            try:
                threshold = float(input("Similarity threshold (0.5-0.9, default 0.7): ") or "0.7")
                similarity_threshold = max(0.5, min(0.9, threshold))
            except ValueError:
                similarity_threshold = 0.7
            
            # Preview similar tokens for first token only (to avoid spam)
            if tokens:
                print(f"Preview for '{tokens[0]}':")
                similar_tokens = self.get_similar_tokens_by_embedding(tokens[0], similarity_threshold, max_tokens=5)
                
                total_estimated = len(tokens) * len(similar_tokens)
                confirm = input(f"\nEstimated {total_estimated} total tokens to modify. Continue? (y/N): ")
                if not confirm.lower().startswith('y'):
                    return
        
        # Get modification type
        print("\nModification types:")
        print("1. Zero (set to 0)")
        print("2. Reduce (decrease)")
        print("3. Boost (increase)")
        
        mod_choice = input("Choose: ").strip()
        
        if mod_choice == "1":
            mod_type = "zero"
            factor = 0.0
        elif mod_choice in ["2", "3"]:
            mod_type = "reduce" if mod_choice == "2" else "boost"
            try:
                factor = float(input(f"Factor (0.0-1.0): "))
                factor = max(0.0, min(1.0, factor))
            except ValueError:
                print("Invalid factor")
                return
        else:
            print("Invalid option")
            return
        
        # Apply modifications to all tokens
        print(f"\n🔧 Applying {mod_type} modifications...")
        for i, token in enumerate(tokens, 1):
            print(f"\n--- Processing token {i}/{len(tokens)}: '{token}' ---")
            self.modify_token_weights(token, mod_type, factor, include_variants, use_semantic, similarity_threshold)
        
        print(f"\n✅ Completed modifications for all {len(tokens)} tokens!")

    def show_modifications_table(self):
        """Show table of applied modifications"""
        if not self.modifications:
            print("No modifications applied to model")
            return
        
        print("\n" + "="*80)
        print("APPLIED MODIFICATIONS OVERVIEW")
        print("="*80)
        print(f"{'Token':<15} {'Layer':<25} {'Type':<10} {'Original':<10} {'New':<10} {'Factor':<10}")
        print("-"*80)
        
        for mod in self.modifications:
            print(f"{mod.token:<15} {mod.layer_name[-25:]:<25} {mod.modification_type:<10} "
                  f"{mod.original_value:<10.4f} {mod.new_value:<10.4f} {mod.factor:<10.2f}")
        
        if self.system_prompt and self.system_prompt != self.original_system_prompt:
            print("-"*80)
            print("MODIFIED SYSTEM PROMPT:")
            print(f"Original: {self.original_system_prompt[:50]}...")
            print(f"Current:  {self.system_prompt[:50]}...")
        
        print("="*80)

    def handle_category_modification(self):
        """Handle category-based token modification"""
        print("Available categories:")
        for i, (category, tokens) in enumerate(self.useful_tokens.items(), 1):
            total_variants = len(self.apply_to_all_variants(tokens))
            print(f"{i}. {category.title()} ({len(tokens)} base → {total_variants} with variants)")
        print("0. ALL CATEGORIES")
        
        try:
            cat_choice = input("Choose category: ").strip()
            
            if cat_choice == "0":
                # Apply to all categories with variants
                tokens_to_process = []
                for category_tokens in self.useful_tokens.values():
                    tokens_to_process.extend(category_tokens)
                tokens_to_process = self.apply_to_all_variants(tokens_to_process)
                print(f"Selected {len(tokens_to_process)} tokens (all variants included)")
            
            else:
                cat_idx = int(cat_choice) - 1
                category = list(self.useful_tokens.keys())[cat_idx]
                base_tokens = self.useful_tokens[category]
                
                print(f"Base tokens in '{category}': {base_tokens}")
                
                include_all_variants = input("Include all variants automatically? (Y/n): ")
                if not include_all_variants.lower().startswith('n'):
                    tokens_to_process = self.apply_to_all_variants(base_tokens)
                    print(f"Included all variants: {len(tokens_to_process)} total tokens")
                else:
                    tokens_to_process = base_tokens.copy()
                    print(f"Selected {len(tokens_to_process)} base tokens")
            
        except (ValueError, IndexError):
            print("Invalid selection")
            return
        
        # Get modification type
        print("\nModification types:")
        print("1. Zero (set to 0)")
        print("2. Reduce (decrease)")
        print("3. Boost (increase)")
        
        mod_choice = input("Choose: ").strip()
        
        if mod_choice == "1":
            mod_type = "zero"
            factor = 0.0
        elif mod_choice in ["2", "3"]:
            mod_type = "reduce" if mod_choice == "2" else "boost"
            try:
                factor = float(input(f"Factor (0.0-1.0): "))
                factor = max(0.0, min(1.0, factor))
            except ValueError:
                print("Invalid factor")
                return
        else:
            print("Invalid option")
            return
        
        # Confirm for large operations
        if len(tokens_to_process) > 10:
            confirm = input(f"About to modify {len(tokens_to_process)} tokens. Continue? (y/N): ")
            if not confirm.lower().startswith('y'):
                print("Operation cancelled")
                return
        
        # Apply modifications
        print(f"\nApplying modifications to {len(tokens_to_process)} tokens...")
        for i, token in enumerate(tokens_to_process, 1):
            print(f"Processing {i}/{len(tokens_to_process)}: {token}")
            self.modify_token_weights(token, mod_type, factor, False)

    def save_modifications_config(self):
        """Save modifications configuration"""
        filename = input("Config filename (modifications.json): ").strip() or "modifications.json"
        
        config = {
            "model_name": self.model_name,
            "modifications": [
                {
                    "token": mod.token,
                    "modification_type": mod.modification_type,
                    "factor": mod.factor
                } for mod in self.modifications
            ],
            "system_prompt": self.system_prompt,
            "original_system_prompt": self.original_system_prompt
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration saved to {filename}")
        except Exception as e:
            print(f"❌ Save error: {e}")

    def load_modifications_config(self):
        """Load modifications configuration"""
        filename = input("Config filename (modifications.json): ").strip() or "modifications.json"
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"✅ Configuration loaded from {filename}")
            print(f"Model: {config.get('model_name', 'N/A')}")
            print(f"Modifications: {len(config.get('modifications', []))}")
            
            apply = input("Apply configuration? (y/N): ").lower().startswith('y')
            if apply and self.model:
                # Reapply modifications
                for mod_config in config.get('modifications', []):
                    self.modify_token_weights(
                        mod_config['token'],
                        mod_config['modification_type'],
                        mod_config['factor'],
                        False
                    )
                
                if config.get('system_prompt'):
                    self.set_system_prompt(config['system_prompt'])
                
                print("✅ Configuration applied")
        
        except Exception as e:
            print(f"❌ Load error: {e}")

    def chat_loop(self):
        """Interactive chat loop with configurable response length"""
        if not self.model:
            print("❌ No model loaded!")
            return
        
        print(f"\n💬 Chat with {self.model_name}")
        print("Commands: /system <prompt> | /weights | /mods | /settings | /length <num> | /quit | /help")
        if self.modifications:
            print(f"🔧 Active modifications: {len(self.modifications)}")
        print(f"⚙️ Response length: {self.generation_settings['max_new_tokens']} tokens")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/quit"):
                    print("👋 Goodbye!")
                    break
                elif user_input.startswith("/help"):
                    print("Available commands:")
                    print("  /system <prompt> - Set system prompt")
                    print("  /weights - Open weight modification menu")
                    print("  /mods - Show current modifications")
                    print("  /settings - Generation settings menu")
                    print("  /length <number> - Set response length (tokens)")
                    print("  /quit - Exit chat")
                    print("  /help - Show this help")
                    continue
                elif user_input.startswith("/system "):
                    new_prompt = user_input[8:].strip()
                    if new_prompt:
                        self.set_system_prompt(new_prompt)
                    else:
                        print(f"Current system prompt: {self.system_prompt}")
                    continue
                elif user_input.startswith("/length "):
                    try:
                        new_length = int(user_input[8:].strip())
                        if 10 <= new_length <= 2000:
                            self.generation_settings['max_new_tokens'] = new_length
                            print(f"✅ Response length set to {new_length} tokens")
                        else:
                            print("❌ Length must be between 10 and 2000 tokens")
                    except ValueError:
                        print("❌ Invalid number")
                    continue
                elif user_input.startswith("/settings"):
                    self.generation_settings_menu()
                    continue
                elif user_input.startswith("/weights"):
                    self.weight_modification_menu()
                    continue
                elif user_input.startswith("/mods"):
                    self.show_modifications_table()
                    continue
                
                # Generate response
                print("🤖 AI: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_main_menu(self):
        """Show main application menu"""
        print("\n🚀 Advanced Model Weight Manipulator")
        print("=" * 50)
        
        if self.model:
            print(f"📊 Current model: {self.model_name}")
            print(f"🔧 Modifications: {len(self.modifications)}")
            if self.system_prompt != self.original_system_prompt:
                print(f"💬 System prompt: Custom ({len(self.system_prompt)} chars)")
            else:
                print("💬 System prompt: Default")
        else:
            print("📊 No model loaded")
        
        print("\nOptions:")
        print("1. Load/Download model")
        print("2. Chat with model")
        print("3. Modify weights")
        print("4. System prompt settings")
        print("5. View modifications")
        print("6. Save/Load config")
        print("0. Exit")
        print("-" * 50)

    def handle_model_loading(self):
        """Handle model loading/downloading"""
        # Check for local models
        local_models = self.scan_local_models()
        popular_models = self.list_popular_models()
        
        print("\nModel loading options:")
        print("1. Load local model")
        print("2. Download popular model")
        print("3. Download custom model")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            if not local_models:
                print("❌ No local models found!")
                return
            
            print("Available local models:")
            for i, model in enumerate(local_models, 1):
                print(f"  {i}. {model}")
            
            try:
                idx = int(input("Select model: ")) - 1
                if 0 <= idx < len(local_models):
                    model_path = local_models[idx]
                    use_quant = input("Use 8-bit quantization? (y/N): ").lower().startswith('y')
                    return self.load_model(model_path, use_quant)
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Invalid input!")
        
        elif choice == "2":
            print("Popular models:")
            for i, model in enumerate(popular_models[:15], 1):  # Show first 15
                print(f"  {i:2d}. {model}")
            
            if len(popular_models) > 15:
                print(f"  ... and {len(popular_models) - 15} more")
                show_all = input("Show all models? (y/N): ").lower().startswith('y')
                if show_all:
                    for i, model in enumerate(popular_models[15:], 16):
                        print(f"  {i:2d}. {model}")
            
            try:
                idx = int(input("Select model: ")) - 1
                if 0 <= idx < len(popular_models):
                    model_name = popular_models[idx]
                    use_quant = input("Use 8-bit quantization? (y/N): ").lower().startswith('y')
                    return self.load_model(model_name, use_quant)
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Invalid input!")
        
        elif choice == "3":
            model_name = input("Enter HuggingFace model name: ").strip()
            if model_name:
                use_quant = input("Use 8-bit quantization? (y/N): ").lower().startswith('y')
                return self.load_model(model_name, use_quant)
        
        return False

    def handle_system_prompt_settings(self):
        """Handle system prompt configuration"""
        print("\n--- SYSTEM PROMPT SETTINGS ---")
        
        if self.system_prompt:
            print(f"Current prompt: {self.system_prompt[:100]}...")
        
        print("Options:")
        print("1. Set new prompt")
        print("2. Edit current prompt")
        print("3. Reset to default")
        print("4. Load prompt presets")
        
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            prompt = input("Enter new system prompt: ").strip()
            if prompt:
                self.set_system_prompt(prompt)
        
        elif choice == "2":
            print(f"Current prompt:\n{self.system_prompt}")
            prompt = input("Enter new prompt: ").strip()
            if prompt:
                self.set_system_prompt(prompt)
        
        elif choice == "3":
            self.set_system_prompt(self.original_system_prompt or "You are a helpful AI assistant.")
            print("✅ Reset to default prompt")
        
        elif choice == "4":
            self.show_prompt_presets()

    def show_prompt_presets(self):
        """Show and apply prompt presets"""
        presets = {
            "helpful": "You are a helpful AI assistant.",
            "creative": "You are a creative and imaginative AI assistant who loves to explore new ideas.",
            "analytical": "You are an analytical AI assistant who provides detailed, logical responses.",
            "concise": "You are a concise AI assistant who gives brief, direct answers.",
            "educational": "You are an educational AI assistant who explains concepts clearly and thoroughly.",
            "coding": "You are a coding AI assistant who provides clean, well-documented code solutions.",
            "research": "You are a research AI assistant who provides comprehensive, well-sourced information."
        }
        
        print("Available presets:")
        for i, (name, prompt) in enumerate(presets.items(), 1):
            print(f"  {i}. {name.title()}: {prompt[:60]}...")
        
        try:
            idx = int(input("Select preset (0 for custom): ")) - 1
            if idx == -1:
                custom = input("Enter custom prompt: ").strip()
                if custom:
                    self.set_system_prompt(custom)
            elif 0 <= idx < len(presets):
                preset_name = list(presets.keys())[idx]
                self.set_system_prompt(presets[preset_name])
            else:
                print("❌ Invalid selection!")
        except ValueError:
            print("❌ Invalid input!")

    def handle_config_management(self):
        """Handle configuration save/load"""
        print("\n--- CONFIGURATION MANAGEMENT ---")
        print("1. Save current configuration")
        print("2. Load configuration")
        print("3. List saved configurations")
        
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            self.save_modifications_config()
        elif choice == "2":
            self.load_modifications_config()
        elif choice == "3":
            self.list_saved_configs()

    def list_saved_configs(self):
        """List available configuration files"""
        config_files = list(Path(".").glob("*.json"))
        
        if not config_files:
            print("No configuration files found")
            return
        
        print("Available configuration files:")
        for i, config_file in enumerate(config_files, 1):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_name = config.get('model_name', 'Unknown')
                    mod_count = len(config.get('modifications', []))
                    print(f"  {i}. {config_file.name} - {model_name} ({mod_count} mods)")
            except:
                print(f"  {i}. {config_file.name} - (invalid config)")

    def run(self):
        """Main application loop"""
        print("🚀 Advanced Model Weight Manipulator")
        print("Enhanced version with weight editing capabilities")
        print("=" * 60)
        
        while True:
            try:
                self.show_main_menu()
                choice = input("\nChoose option (0-6): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                
                elif choice == "1":
                    self.handle_model_loading()
                
                elif choice == "2":
                    if not self.model:
                        print("❌ Load a model first!")
                        continue
                    self.chat_loop()
                
                elif choice == "3":
                    if not self.model:
                        print("❌ Load a model first!")
                        continue
                    self.weight_modification_menu()
                
                elif choice == "4":
                    self.handle_system_prompt_settings()
                
                elif choice == "5":
                    self.generation_settings_menu()
                
                elif choice == "6":
                    self.show_modifications_table()
                
                elif choice == "7":
                    self.handle_config_management()
                
                else:
                    print("❌ Invalid option!")
                
                # Pause before showing menu again
                if choice != "0":
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    manager = AdvancedModelManager()
    manager.run()
