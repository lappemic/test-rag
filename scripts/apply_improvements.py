#!/usr/bin/env python3
"""
Comprehensive Script to Apply Vector Storage Improvements
This script orchestrates the entire improvement process with backup and rollback capabilities.
"""
import os
import sys
import time
import shutil
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovementManager:
    """Manages the complete improvement application process."""
    
    def __init__(self):
        self.backup_dir = Path(f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.original_chroma_path = Path("./chroma_db")
        self.data_dir = Path("./data")
        
    def create_backup(self):
        """Create a backup of the current system state."""
        logger.info("Creating backup of current system state...")
        
        try:
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Backup ChromaDB
            if self.original_chroma_path.exists():
                chroma_backup = self.backup_dir / "chroma_db"
                shutil.copytree(self.original_chroma_path, chroma_backup)
                logger.info(f"✅ ChromaDB backed up to {chroma_backup}")
            else:
                logger.warning("No existing ChromaDB found to backup")
            
            # Backup existing chunk files
            chunk_files = list(self.data_dir.glob("*law_chunks*.json"))
            if chunk_files:
                for chunk_file in chunk_files:
                    shutil.copy2(chunk_file, self.backup_dir)
                logger.info(f"✅ Backed up {len(chunk_files)} chunk files")
            
            # Create backup manifest
            manifest = {
                "backup_time": datetime.now().isoformat(),
                "original_chroma_exists": self.original_chroma_path.exists(),
                "chunk_files_backed_up": [f.name for f in chunk_files],
                "backup_dir": str(self.backup_dir)
            }
            
            with open(self.backup_dir / "backup_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Backup completed: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        issues = []
        
        # Check Python scripts exist
        required_scripts = [
            "scripts/generate_chunks_improved_production.py",
            "scripts/load_chunks_to_chromadb_improved.py"
        ]
        
        for script in required_scripts:
            if not Path(script).exists():
                issues.append(f"Missing script: {script}")
        
        # Check XML files exist
        xml_files = list(self.data_dir.glob("*.xml"))
        if not xml_files:
            issues.append("No XML files found in data/ directory")
        else:
            logger.info(f"Found {len(xml_files)} XML files: {[f.name for f in xml_files]}")
        
        # Check environment
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY not set in environment")
        
        # Check dependencies
        try:
            import chromadb
            import langchain_openai
            logger.info("✅ Required packages available")
        except ImportError as e:
            issues.append(f"Missing dependency: {e}")
        
        if issues:
            logger.error("❌ Prerequisites not met:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def run_chunking(self):
        """Run the improved chunking script."""
        logger.info("Running improved chunking script...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                "scripts/generate_chunks_improved_production.py"
            ], capture_output=True, text=True, check=True)
            
            logger.info("✅ Chunking completed successfully")
            logger.info("Output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Chunking failed: {e}")
            logger.error("Error output:")
            for line in e.stderr.split('\n'):
                if line.strip():
                    logger.error(f"  {line}")
            return False
    
    def run_loading(self, force_replace=True):
        """Run the improved loading script."""
        logger.info("Running improved loading script...")
        
        try:
            # Prepare input for the script (auto-select full replacement)
            user_input = "2\n" if force_replace else "1\n"
            
            result = subprocess.run([
                sys.executable, 
                "scripts/load_chunks_to_chromadb_improved.py"
            ], input=user_input, capture_output=True, text=True, check=True)
            
            logger.info("✅ Loading completed successfully")
            logger.info("Output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Loading failed: {e}")
            logger.error("Error output:")
            for line in e.stderr.split('\n'):
                if line.strip():
                    logger.error(f"  {line}")
            return False
    
    def verify_improvements(self):
        """Verify that improvements were applied successfully."""
        logger.info("Verifying improvements...")
        
        try:
            # Check if new chunk files exist
            improved_chunks = list(self.data_dir.glob("*law_chunks_improved.json"))
            if not improved_chunks:
                logger.error("❌ No improved chunk files found")
                return False
            
            latest_improved = max(improved_chunks, key=os.path.getmtime)
            logger.info(f"✅ Found improved chunks: {latest_improved.name}")
            
            # Check chunk file content
            with open(latest_improved, 'r') as f:
                chunks = json.load(f)
            
            # Verify improvements
            cross_ref_chunks = sum(1 for c in chunks if c.get('metadata', {}).get('cross_references'))
            avg_tokens = sum(c.get('metadata', {}).get('estimated_tokens', 0) for c in chunks) / len(chunks)
            
            logger.info(f"✅ Total chunks: {len(chunks)}")
            logger.info(f"✅ Average tokens per chunk: {avg_tokens:.1f}")
            logger.info(f"✅ Chunks with cross-references: {cross_ref_chunks} ({cross_ref_chunks/len(chunks)*100:.1f}%)")
            
            # Check ChromaDB collections
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()
            
            total_count = sum(c.count() for c in collections)
            logger.info(f"✅ ChromaDB collections: {len(collections)}")
            logger.info(f"✅ Total chunks in ChromaDB: {total_count}")
            
            if total_count == 0:
                logger.error("❌ No chunks found in ChromaDB")
                return False
            
            logger.info("✅ Improvements verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def rollback(self):
        """Rollback to the backed up state."""
        logger.info("Rolling back to previous state...")
        
        try:
            if not self.backup_dir.exists():
                logger.error(f"❌ Backup directory not found: {self.backup_dir}")
                return False
            
            # Load backup manifest
            manifest_path = self.backup_dir / "backup_manifest.json"
            if not manifest_path.exists():
                logger.error("❌ Backup manifest not found")
                return False
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Restore ChromaDB
            if manifest.get("original_chroma_exists"):
                if self.original_chroma_path.exists():
                    shutil.rmtree(self.original_chroma_path)
                
                chroma_backup = self.backup_dir / "chroma_db"
                if chroma_backup.exists():
                    shutil.copytree(chroma_backup, self.original_chroma_path)
                    logger.info("✅ ChromaDB restored")
            
            # Restore chunk files (optional - new files don't overwrite old ones)
            logger.info("✅ Rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def cleanup_backup(self):
        """Clean up backup directory after successful application."""
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
                logger.info(f"✅ Cleaned up backup directory: {self.backup_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup backup: {e}")

def main():
    """Main function to orchestrate the improvement application."""
    print("🚀 Vector Storage Improvements Application")
    print("=" * 60)
    
    manager = ImprovementManager()
    
    # Step 1: Check prerequisites
    if not manager.check_prerequisites():
        print("❌ Prerequisites not met. Please fix the issues above and try again.")
        return 1
    
    # Step 2: Create backup
    print("\n📦 Creating backup...")
    if not manager.create_backup():
        print("❌ Backup failed. Aborting to prevent data loss.")
        return 1
    
    # Step 3: Confirm with user
    print(f"\n✅ Backup created: {manager.backup_dir}")
    print("\nThis process will:")
    print("1. Generate improved chunks with semantic chunking and cross-references")
    print("2. Replace your current ChromaDB content with the improved version")
    print("3. Verify that improvements were applied successfully")
    print("\nIf anything goes wrong, you can rollback using the backup.")
    
    while True:
        confirm = input("\nDo you want to proceed? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no']:
            print("❌ Aborted by user")
            return 0
        else:
            print("Please enter 'y' or 'n'")
    
    success = True
    
    try:
        # Step 4: Run chunking
        print("\n🔧 Step 1/3: Generating improved chunks...")
        if not manager.run_chunking():
            success = False
            raise Exception("Chunking failed")
        
        # Step 5: Run loading
        print("\n💾 Step 2/3: Loading improved chunks to ChromaDB...")
        if not manager.run_loading(force_replace=True):
            success = False
            raise Exception("Loading failed")
        
        # Step 6: Verify
        print("\n✅ Step 3/3: Verifying improvements...")
        if not manager.verify_improvements():
            success = False
            raise Exception("Verification failed")
        
        # Success!
        print("\n🎉 IMPROVEMENTS APPLIED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Your vector storage has been improved with:")
        print("  - Semantic chunking with overlap")
        print("  - Cross-reference extraction")
        print("  - Enhanced metadata")
        print("  - Quality validation")
        print("\n🔍 Test your system with some queries to see the improvements!")
        
        # Offer to cleanup backup
        while True:
            cleanup = input("\nDelete backup files? (y/n): ").strip().lower()
            if cleanup in ['y', 'yes']:
                manager.cleanup_backup()
                break
            elif cleanup in ['n', 'no']:
                print(f"📦 Backup preserved at: {manager.backup_dir}")
                break
            else:
                print("Please enter 'y' or 'n'")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 60)
        
        # Offer rollback
        while True:
            rollback = input("\nDo you want to rollback to the previous state? (y/n): ").strip().lower()
            if rollback in ['y', 'yes']:
                print("\n🔄 Rolling back...")
                if manager.rollback():
                    print("✅ Rollback completed successfully")
                else:
                    print("❌ Rollback failed - you may need to restore manually")
                break
            elif rollback in ['n', 'no']:
                print(f"📦 Backup available at: {manager.backup_dir}")
                break
            else:
                print("Please enter 'y' or 'n'")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 