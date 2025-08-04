import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAhpZjC8uHToOaaUa4qwMDUOhP5Se0cqsY")

# List all available models
for model in genai.list_models():
    print(f"{model.name} ➡️ supports: {model.supported_generation_methods}")
