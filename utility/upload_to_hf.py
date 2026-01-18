
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

# --- CONFIGURATION ---
HF_USERNAME = 'christo357' # 
REPO_NAME = "TQC_FetchPickAndPlace_v4" # Name of your repo

# List of files to upload
FILES_TO_UPLOAD = [
    "./../models/tqcdense_model.zip",
    "./../models/tqcdense_vecnorm.pkl"
]
def upload_models():
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    print(f"🚀 Preparing to upload to {repo_id}...")

    # 1. Create Repo (if it doesn't exist)
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print("✅ Repo checked/created.")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return

    # 2. Upload Files
    for file_path in FILES_TO_UPLOAD:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        file_name = os.path.basename(file_path)
        print(f"📤 Uploading {file_name}...")
        
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name, # Save at root of repo
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"   ✅ {file_name} uploaded successfully.")
        except Exception as e:
            print(f"   ❌ Failed to upload {file_name}: {e}")

    print("\n🎉 Upload Complete! Your model is hosted.")

if __name__ == "__main__":
    upload_models()