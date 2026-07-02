"""
download_docs.py

What this file does:
Downloads real technical documentation from GitHub for:
1. FastAPI - a popular Python web framework
2. Kubernetes - the container orchestration platform

Why real docs?
Because our AI assistant needs actual knowledge to answer questions like
"How do I fix a CrashLoopBackOff?" - it learns from these real documents.

These docs become the UNSTRUCTURED data source in our project.
Tickets = structured (rows and columns in a database)
Docs = unstructured (paragraphs of text)
The AI agent searches BOTH depending on what you ask.
"""

import os
import subprocess
import shutil
import stat


def remove_readonly(func, path, _):
    """
    Windows sometimes makes git files read-only.
    This function forces them to be deletable.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_remove(path):
    """Safely remove a folder even if Windows locked some files"""
    if os.path.exists(path):
        shutil.rmtree(path, onexc=remove_readonly)


def download_fastapi_docs():
    """
    Downloads FastAPI's official English documentation from GitHub.
    These are the same docs you see on fastapi.tiangolo.com
    """
    print("📥 Downloading FastAPI documentation...")

    # Where we'll save the docs
    output_dir = "data/raw/docs/fastapi"
    os.makedirs(output_dir, exist_ok=True)

    # We use git sparse-checkout to download ONLY the docs folder
    # instead of the entire FastAPI codebase (saves time and space)
    temp_dir = "data/raw/temp_fastapi"

    try:
        # Clone only the structure first (no actual files yet)
        subprocess.run([
            "git", "clone",
            "--depth", "1",           # only latest version, not full history
            "--filter=blob:none",     # don't download file contents yet
            "--sparse",               # sparse mode = selective download
            "https://github.com/fastapi/fastapi.git",
            temp_dir
        ], check=True, capture_output=True)

        # Now tell git we only want the English docs folder
        subprocess.run([
            "git", "sparse-checkout", "set", "docs/en/docs"
        ], check=True, capture_output=True, cwd=temp_dir)

        # Copy only the markdown files to our clean output folder
        docs_path = os.path.join(temp_dir, "docs", "en", "docs")
        copied = 0
        for root, dirs, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".md"):
                    src = os.path.join(root, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy2(src, dst)
                    copied += 1

        print(f"   ✅ FastAPI: {copied} documentation files downloaded")

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error downloading FastAPI docs: {e}")

    finally:
        # Clean up the temporary git folder
        safe_remove(temp_dir)


def download_kubernetes_docs():
    """
    Downloads Kubernetes official documentation from GitHub.
    Specifically the concepts and tasks sections - most useful for our use case.
    """
    print("📥 Downloading Kubernetes documentation...")

    output_dir = "data/raw/docs/kubernetes"
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = "data/raw/temp_kubernetes"

    try:
        subprocess.run([
            "git", "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/kubernetes/website.git",
            temp_dir
        ], check=True, capture_output=True)

        # We want concepts (what things are) and tasks (how to do things)
        subprocess.run([
            "git", "sparse-checkout", "set",
            "content/en/docs/concepts",
            "content/en/docs/tasks"
        ], check=True, capture_output=True, cwd=temp_dir)

        # Copy markdown files
        docs_path = os.path.join(temp_dir, "content", "en", "docs")
        copied = 0
        for root, dirs, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".md"):
                    src = os.path.join(root, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy2(src, dst)
                    copied += 1

        print(f"   ✅ Kubernetes: {copied} documentation files downloaded")

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error downloading Kubernetes docs: {e}")

    finally:
        safe_remove(temp_dir)


def verify_downloads():
    """
    Checks that everything downloaded correctly and shows a summary
    """
    print("\n📊 Download Summary:")

    fastapi_count = 0
    k8s_count = 0

    fastapi_path = "data/raw/docs/fastapi"
    k8s_path = "data/raw/docs/kubernetes"

    if os.path.exists(fastapi_path):
        fastapi_count = len([f for f in os.listdir(fastapi_path)
                             if f.endswith(".md")])
        print(f"   FastAPI docs:     {fastapi_count} files")
    else:
        print("   FastAPI docs:     ❌ folder not found")

    if os.path.exists(k8s_path):
        k8s_count = len([f for f in os.listdir(k8s_path)
                         if f.endswith(".md")])
        print(f"   Kubernetes docs:  {k8s_count} files")
    else:
        print("   Kubernetes docs:  ❌ folder not found")

    total = fastapi_count + k8s_count
    print(f"   Total:            {total} documentation files")
    print(f"\n✅ Docs saved to data/raw/docs/")
    print("🚀 Ready for next step: chunking and embedding!")


def main():
    print("🚀 Starting documentation download...\n")
    download_fastapi_docs()
    download_kubernetes_docs()
    verify_downloads()


if __name__ == "__main__":
    main()