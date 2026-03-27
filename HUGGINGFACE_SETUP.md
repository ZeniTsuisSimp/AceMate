# Deploying to Hugging Face Spaces

Since your trial for Railway is over, Hugging Face Spaces is a great free alternative to host your Streamlit application.

Here is a step-by-step guide to get your application up and running on Hugging Face Spaces.

## Step 1: Create a Hugging Face Account
If you don't already have one, sign up for a free account at [Hugging Face](https://huggingface.co/join).

## Step 2: Create a New Space
1. Go to your [Hugging Face Profile](https://huggingface.co/) and click on your profile picture in the top right corner.
2. Select **New Space**.
3. **Space Name**: Give it a name like `AceMate`.
4. **License**: Choose an appropriate license (e.g., `MIT` or `OpenRAIL`).
5. **Select the Space SDK**: Choose **Streamlit**.
6. **Space Hardware**: The default `Free` tier (CPU basic, 16GB RAM) will be selected. You can upgrade later if you need more power for the embedding models.
7. Click **Create Space**.

## Step 3: Add Your Secrets
Before uploading the code, add your API keys to the Space Secrets.
1. In your newly created Space, go to the **Settings** tab.
2. Scroll down to the **Variables and secrets** section.
3. Click on **New secret** for each environment variable present in your `.env` file. You need to add:
   - `SARVAM_API_KEY`: Your Sarvam API Key
   - `QDRANT_URL`: URL to your Qdrant cluster (if you are using cloud instance)
   - `QDRANT_API_KEY`: Your Qdrant API Key (if you are using cloud instance)
   *(Add any other keys you may have in your .env file)*

## Step 4: Upload Your Code
You can upload the code directly via the browser or using Git.

### Method A: Direct Upload (Easiest)
1. Go to the **Files** tab of your Space.
2. Click **Add file** > **Upload files**.
3. Drag and drop all the project files from `d:\AceMate\examprep-ai` (excluding `.git`, `__pycache__`, `.env`, and `venv` folders) into the upload area.
4. **Important**: Make sure `app.py`, `requirements.txt`, and `packages.txt` are at the root level.
5. Add a commit message (e.g., "Initial commit") and click **Commit changes to main**.

### Method B: Using Git
1. Get the cloning URL for your Space (e.g., `https://huggingface.co/spaces/YOUR_USERNAME/Acemate`).
2. Run the following commands in your terminal:
   ```bash
   # Add the Hugging Face space as a git remote
   git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/Acemate
   
   # Push your code to the Hugging Face main branch
   git push huggingface main
   ```
   *Note: You may be prompted for your Hugging Face username and a User Access Token (which you can generate in your Hugging Face Settings -> Access Tokens).*

## Step 5: Wait for the Build
Once the files are pushed, Hugging Face will automatically:
1. Install system dependencies from `packages.txt`.
2. Install Python packages from `requirements.txt`.
3. Start the Streamlit app.

You can monitor the build progress by clicking on the **Logs** button near the top right of your Space.

Once it's done building, it will display **Running** and your app will be accessible!
