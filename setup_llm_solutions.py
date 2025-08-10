#!/usr/bin/env python3
"""Setup script for LLM solution generation."""

import subprocess
import sys
import os

def install_requests():
    """Install requests library if not available."""
    try:
        import requests
        print("✅ requests library already installed")
        return True
    except ImportError:
        print("📦 Installing requests library...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
            print("✅ Successfully installed requests")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requests: {e}")
            return False

def test_with_api_key():
    """Test solution generation with API key if available."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ℹ️  To enable LLM solution generation, set your OpenAI API key:")
        print("   export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print("✅ OpenAI API key found")
    
    # Test solution generation
    try:
        from mathematics_dataset.solution_generator import generate_solution
        
        question = "Mental math: approximately 47 - 820?"
        exact_answer = -773.0
        approx_answer = "-770"
        
        print(f"\n🧮 Testing LLM solution generation:")
        print(f"Question: {question}")
        
        solution = generate_solution(question, exact_answer, approx_answer)
        print(f"Solution:\n{solution}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing solution generation: {e}")
        return False

def main():
    print("🚀 Setting up LLM solution generation for street_math dataset")
    
    # Install requests
    if not install_requests():
        print("❌ Setup failed: Could not install requests library")
        sys.exit(1)
    
    # Test with API key
    if test_with_api_key():
        print("\n✅ LLM solution generation is ready!")
        print("\n📋 Usage:")
        print("   GENERATE_SOLUTIONS=true python -m mathematics_dataset.generate_to_file \\")
        print("     --filter=street_math --per_train_module=100 \\") 
        print("     --output_dir=./street_math_dataset --format=json")
    else:
        print("\n⚠️  LLM solution generation not configured")
        print("   Basic solution generation will be used instead")
        
    print("\n📝 The generated dataset will include:")
    print("   • input: The math question")
    print("   • approximate_answer: Mental math approximation")
    print("   • exact_answer: Precise mathematical result")
    print("   • lower_bound: 10% below exact answer")
    print("   • upper_bound: 10% above exact answer")
    print("   • solution: Step-by-step solution (LLM or basic)")

if __name__ == "__main__":
    main()