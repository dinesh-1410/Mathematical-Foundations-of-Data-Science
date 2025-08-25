# 🚀 GitHub Repository Setup Guide

## Quick Start Instructions

### Step 1: Create GitHub Repository
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name:** `Mathematical-Foundations-of-Data-Science`
   - **Description:** `Mathematical Foundations of Data Science (MFDS) course implementations from IIT Madras`
   - **Visibility:** Choose **Public**
   - **DO NOT** check "Add a README file"
   - **DO NOT** check "Add .gitignore"
   - **DO NOT** check "Choose a license"
5. Click **"Create repository"**

### Step 2: Initialize Local Repository
Open your terminal/command prompt and run:

```bash
# Navigate to your MFDS folder
cd "C:\@D\SEM VI\MFDS"

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: MFDS course repository - Linear Regression Models and Lomb-Scargle Periodogram"
```

### Step 3: Connect to GitHub
```bash
# Add remote origin (replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Mathematical-Foundations-of-Data-Science.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 4: Verify Setup
1. Go to your GitHub profile
2. You should see the new repository listed
3. Click on it to verify the README renders correctly
4. Check that all files are uploaded

## Repository Structure Verification

Your repository should now contain:
- ✅ `README.md` - Main documentation
- ✅ `Project/` - Core implementations
- ✅ `Assignments/` - Course assignments
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Git exclusions
- ✅ All other course materials

## Next Steps

1. **Pin the Repository:** Go to your GitHub profile and pin this repository
2. **Add Topics:** Add relevant topics to make it discoverable
3. **Share:** Include the repository link in your resume and LinkedIn profile
4. **Update Regularly:** Commit improvements and new features

## Repository URL Format
Your repository will be available at:
```
https://github.com/YOUR_USERNAME/Mathematical-Foundations-of-Data-Science
```

## Need Help?
- Check GitHub's [Getting Started Guide](https://docs.github.com/en/get-started)
- Review [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- Contact GitHub Support if you encounter issues

---

**Congratulations!** You now have a professional portfolio repository showcasing your mathematical foundations in data science! 🎉
