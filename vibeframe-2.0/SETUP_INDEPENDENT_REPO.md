# 🚀 Setting Up VibeFrame 2.0 as an Independent Repository

Follow these steps to create VibeFrame 2.0 as a standalone project separate from Vibehub.

---

## Option 1: Automated Setup (Recommended)

### Step 1: Run the Setup Script

From the **parent directory** of Vibehub- (not inside ai_mv_generator):

```bash
cd /Users/puiyuenwong/Desktop/Real\ uni\ \(1\)/PCLL\ 2023\ /Haldanes\ demo\ try\ /Album\ /Apps\ /Vibehub-/

# Run the setup script
bash ai_mv_generator/create_independent_repo.sh
```

This will:
- ✅ Create a new `vibeframe-2.0` directory
- ✅ Initialize git
- ✅ Copy all VibeFrame files
- ✅ Create initial commit

### Step 2: Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `vibeframe-2.0`
3. Description: "AI Music Video Generator - Transform audio into visually consistent music videos"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 3: Push to GitHub

```bash
cd ../vibeframe-2.0

# Add remote
git remote add origin https://github.com/Readyyeahgoooo/vibeframe-2.0.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Option 2: Manual Setup

### Step 1: Create New Directory

```bash
# Navigate to parent directory
cd /Users/puiyuenwong/Desktop/Real\ uni\ \(1\)/PCLL\ 2023\ /Haldanes\ demo\ try\ /Album\ /Apps\ /

# Create new directory
mkdir vibeframe-2.0
cd vibeframe-2.0
```

### Step 2: Initialize Git

```bash
git init
```

### Step 3: Copy Files

```bash
# Copy all files from ai_mv_generator
cp -r ../Vibehub-/ai_mv_generator/* .

# Verify files copied
ls -la
```

You should see:
- vibeframe/ (main package)
- tests/ (test suite)
- app_gradio.py
- requirements.txt
- README.md
- LICENSE
- .gitignore
- etc.

### Step 4: Create Initial Commit

```bash
git add -A
git commit -m "Initial commit: VibeFrame 2.0 - AI Music Video Generator"
```

### Step 5: Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `vibeframe-2.0`
3. Make it Public or Private
4. **DO NOT** check any initialization options
5. Click "Create repository"

### Step 6: Push to GitHub

```bash
# Add remote (replace with your actual GitHub URL)
git remote add origin https://github.com/Readyyeahgoooo/vibeframe-2.0.git

# Push
git branch -M main
git push -u origin main
```

---

## Option 3: Using GitHub CLI (Fastest)

If you have GitHub CLI installed:

```bash
# Navigate to parent directory
cd /Users/puiyuenwong/Desktop/Real\ uni\ \(1\)/PCLL\ 2023\ /Haldanes\ demo\ try\ /Album\ /Apps\ /

# Create new directory
mkdir vibeframe-2.0
cd vibeframe-2.0

# Initialize git
git init

# Copy files
cp -r ../Vibehub-/ai_mv_generator/* .

# Create initial commit
git add -A
git commit -m "Initial commit: VibeFrame 2.0"

# Create GitHub repo and push (all in one command!)
gh repo create vibeframe-2.0 --public --source=. --remote=origin --push
```

---

## Verification

After setup, verify your new repository:

### 1. Check Local Repository

```bash
cd vibeframe-2.0
git status
git log --oneline
```

### 2. Check GitHub

Visit: https://github.com/Readyyeahgoooo/vibeframe-2.0

You should see:
- ✅ README.md displayed
- ✅ All files and folders
- ✅ Commit history
- ✅ LICENSE file

### 3. Test the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app_gradio.py
```

---

## What Gets Copied

The independent repository will include:

### Core Package
- `vibeframe/` - All 11 modules
  - audio_analyzer.py
  - scene_planner.py
  - character_manager.py
  - video_generator.py
  - video_compositor.py
  - project_manager.py
  - workflow.py
  - config.py
  - error_handler.py
  - api_clients.py
  - models.py

### Tests
- `tests/` - Complete test suite (211 tests)
  - test_audio_analyzer.py
  - test_scene_planner.py
  - test_character_manager.py
  - test_video_generator.py
  - test_video_compositor.py
  - test_project_manager.py
  - test_models.py
  - conftest.py

### Documentation
- README.md - Comprehensive guide
- COMPLETION_SUMMARY.md - Project status
- USAGE_GUIDE.md - Usage examples
- LICENSE - MIT License

### Configuration
- requirements.txt - All dependencies
- .env.example - Environment template
- .gitignore - Git exclusions
- app_gradio.py - Web interface

### Specifications
- `.kiro/specs/` - Design documents
  - requirements.md
  - design.md
  - tasks.md

---

## Benefits of Independent Repository

### ✅ Advantages

1. **Clean Project Structure**
   - Standalone project
   - No parent dependencies
   - Clear purpose

2. **Better Discoverability**
   - Easier to find on GitHub
   - Clear project name
   - Focused README

3. **Independent Development**
   - Separate issues/PRs
   - Own release cycle
   - Independent versioning

4. **Easier Collaboration**
   - Contributors focus on one project
   - Clearer contribution guidelines
   - Simpler CI/CD setup

5. **Better Documentation**
   - Project-specific docs
   - Focused examples
   - Clearer purpose

---

## Maintaining Both Repositories

If you want to keep both:

### Vibehub Repository
- Keep as parent/umbrella project
- Link to VibeFrame 2.0 in README
- Use for other Vibehub components

### VibeFrame 2.0 Repository
- Standalone music video generator
- Independent development
- Own issues and releases

### Linking Them

In Vibehub README, add:

```markdown
## Projects

- [VibeFrame 2.0](https://github.com/Readyyeahgoooo/vibeframe-2.0) - AI Music Video Generator
```

---

## Troubleshooting

### Issue: "Repository already exists"

If the GitHub repo name is taken:
```bash
# Use a different name
git remote add origin https://github.com/Readyyeahgoooo/vibeframe-ai.git
```

### Issue: "Permission denied"

Make sure you're authenticated:
```bash
# Check authentication
gh auth status

# Or use SSH
git remote set-url origin git@github.com:Readyyeahgoooo/vibeframe-2.0.git
```

### Issue: "Files not copied"

Verify source path:
```bash
ls -la ../Vibehub-/ai_mv_generator/
```

---

## Next Steps After Setup

1. **Update README**
   - Change repository URLs
   - Update installation instructions
   - Add badges

2. **Set Up GitHub Actions** (Optional)
   - Automated testing
   - Code quality checks
   - Deployment

3. **Create Releases**
   - Tag version 2.0.0
   - Create release notes
   - Publish to PyPI (optional)

4. **Add Topics on GitHub**
   - ai
   - music-video
   - video-generation
   - gradio
   - python

---

## Summary

After following these steps, you'll have:

✅ Independent VibeFrame 2.0 repository
✅ Clean project structure
✅ All code and documentation
✅ Ready for collaboration
✅ Easy to discover and use

**Repository URL:** https://github.com/Readyyeahgoooo/vibeframe-2.0

---

*Need help? Open an issue on GitHub!*
