"""
Deploy causal-ml-skills to Claude Code skills directory.

Usage:
    python deploy.py                    # Deploy all skills (interactive)
    python deploy.py --dry-run          # Preview deployment
    python deploy.py --skill estimator-did  # Deploy specific skill
    python deploy.py --batch            # Deploy all skills without prompts
    python deploy.py --category classic-methods  # Deploy skills in category
    python deploy.py --validate         # Validate before deploying
    python deploy.py --backup           # Create backup before deploy
    python deploy.py --manifest         # Generate deployment manifest
    python deploy.py --version 1.2.0    # Set version for deployment
    python deploy.py --rollback         # Rollback to previous version

Examples:
    python deploy.py --batch --validate --backup
    python deploy.py --category causal-ml --validate
    python deploy.py --skill estimator-did --version 2.0.0 --backup
"""

import hashlib
import json
import shutil
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

SOURCE_DIR = Path(__file__).parent / "skills"
TARGET_DIR = Path(r"C:\Users\tangj\.claude\skills")
BACKUP_DIR = Path(r"C:\Users\tangj\.claude\skills_backups")
MANIFEST_FILE = Path(__file__).parent / "deployment_manifest.json"
VALIDATE_SCRIPT = Path(__file__).parent / "scripts" / "validate_skill.py"

# Default version if not specified
DEFAULT_VERSION = "1.0.0"

# Skill directories organized by category
SKILL_CATEGORIES = {
    "infrastructure": [
        "infrastructure/setup-causal-ml-env",
        "infrastructure/scientific-writing-econ",
    ],
    "classic-methods": [
        "classic-methods/causal-concept-guide",
        "classic-methods/estimator-did",
        "classic-methods/estimator-rd",
        "classic-methods/estimator-iv",
        "classic-methods/estimator-psm",
        "classic-methods/discrete-choice-models",
        "classic-methods/panel-data-models",
        "classic-methods/time-series-econometrics",
    ],
    "ml-foundation": [
        "ml-foundation/ml-preprocessing",
        "ml-foundation/ml-model-linear",
        "ml-foundation/ml-model-tree",
        "ml-foundation/ml-model-advanced",
        "ml-foundation/statistical-analysis",
        "ml-foundation/econometric-eda",
    ],
    "causal-ml": [
        "causal-ml/causal-ddml",
        "causal-ml/causal-forest",
        "causal-ml/causal-mediation-ml",
        "causal-ml/paper-replication-workflow",
        "causal-ml/bayesian-econometrics",
    ],
}

# All skill directories (flattened from categories)
SKILL_DIRS = []
for category_skills in SKILL_CATEGORIES.values():
    SKILL_DIRS.extend(category_skills)


# =============================================================================
# Progress Reporting
# =============================================================================

class ProgressReporter:
    """Handle progress reporting with visual feedback."""

    def __init__(self, total: int, verbose: bool = True):
        self.total = total
        self.current = 0
        self.verbose = verbose
        self.start_time = datetime.now()

    def update(self, message: str, status: str = "info") -> None:
        """Update progress with a message."""
        self.current += 1
        if self.verbose:
            prefix = self._get_status_prefix(status)
            progress = f"[{self.current}/{self.total}]"
            print(f"{prefix} {progress} {message}")

    def _get_status_prefix(self, status: str) -> str:
        """Get colored prefix for status."""
        prefixes = {
            "success": "\033[92m[OK]\033[0m",
            "error": "\033[91m[FAIL]\033[0m",
            "warning": "\033[93m[WARN]\033[0m",
            "info": "\033[94m[INFO]\033[0m",
            "skip": "\033[90m[SKIP]\033[0m",
        }
        return prefixes.get(status, "[INFO]")

    def summary(self, success: int, failed: int, skipped: int = 0) -> None:
        """Print deployment summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print()
        print("=" * 60)
        print("DEPLOYMENT SUMMARY")
        print("=" * 60)
        print(f"Total skills:   {self.total}")
        print(f"Successful:     \033[92m{success}\033[0m")
        print(f"Failed:         \033[91m{failed}\033[0m")
        if skipped > 0:
            print(f"Skipped:        \033[90m{skipped}\033[0m")
        print(f"Time elapsed:   {elapsed:.2f}s")
        print("=" * 60)


# =============================================================================
# File Utilities
# =============================================================================

def get_skill_name(skill_path: str) -> str:
    """Extract skill name from path."""
    return skill_path.split("/")[-1]


def get_skill_category(skill_path: str) -> str:
    """Extract category from skill path."""
    return skill_path.split("/")[0]


def calculate_checksum(directory: Path) -> str:
    """Calculate MD5 checksum for a directory."""
    checksums = []
    if directory.exists():
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                with open(file, "rb") as f:
                    checksums.append(hashlib.md5(f.read()).hexdigest())
    combined = "".join(checksums)
    return hashlib.md5(combined.encode()).hexdigest()


def get_files_info(directory: Path) -> Dict[str, str]:
    """Get file checksums for a directory."""
    files_info = {}
    if directory.exists():
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                rel_path = file.relative_to(directory)
                with open(file, "rb") as f:
                    files_info[str(rel_path)] = hashlib.md5(f.read()).hexdigest()
    return files_info


# =============================================================================
# Manifest Management
# =============================================================================

def load_manifest() -> Dict:
    """Load deployment manifest from file."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": "1.0.0",
        "last_deployment": None,
        "skills": {},
        "history": []
    }


def save_manifest(manifest: Dict) -> None:
    """Save deployment manifest to file."""
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def update_manifest(
    manifest: Dict,
    skill_name: str,
    skill_path: str,
    version: str,
    checksum: str,
    files: Dict[str, str]
) -> None:
    """Update manifest with deployed skill information."""
    now = datetime.now().isoformat()

    # Update skill entry
    previous_version = manifest["skills"].get(skill_name, {}).get("version")
    manifest["skills"][skill_name] = {
        "path": skill_path,
        "version": version,
        "checksum": checksum,
        "files": files,
        "deployed_at": now,
        "previous_version": previous_version,
    }

    # Update global info
    manifest["last_deployment"] = now

    # Add to history
    manifest["history"].append({
        "timestamp": now,
        "action": "deploy",
        "skill": skill_name,
        "version": version,
    })

    # Keep only last 100 history entries
    if len(manifest["history"]) > 100:
        manifest["history"] = manifest["history"][-100:]


# =============================================================================
# Backup and Rollback
# =============================================================================

def create_backup(skill_name: str) -> Optional[Path]:
    """Create backup of existing skill before deployment."""
    source = TARGET_DIR / skill_name
    if not source.exists():
        return None

    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{skill_name}_{timestamp}"

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, backup_path)

    # Keep only last 5 backups per skill
    cleanup_old_backups(skill_name, keep=5)

    return backup_path


def cleanup_old_backups(skill_name: str, keep: int = 5) -> None:
    """Remove old backups, keeping only the most recent ones."""
    if not BACKUP_DIR.exists():
        return

    # Find all backups for this skill
    backups = sorted([
        d for d in BACKUP_DIR.iterdir()
        if d.is_dir() and d.name.startswith(f"{skill_name}_")
    ], key=lambda x: x.name)

    # Remove old backups
    for backup in backups[:-keep]:
        shutil.rmtree(backup)


def get_latest_backup(skill_name: str) -> Optional[Path]:
    """Get the most recent backup for a skill."""
    if not BACKUP_DIR.exists():
        return None

    backups = sorted([
        d for d in BACKUP_DIR.iterdir()
        if d.is_dir() and d.name.startswith(f"{skill_name}_")
    ], key=lambda x: x.name)

    return backups[-1] if backups else None


def rollback_skill(skill_name: str, dry_run: bool = False) -> bool:
    """Rollback a skill to its previous backup."""
    backup = get_latest_backup(skill_name)
    if not backup:
        print(f"  ERROR: No backup found for {skill_name}")
        return False

    target = TARGET_DIR / skill_name

    if dry_run:
        print(f"  [DRY RUN] Would rollback {skill_name} from {backup}")
        return True

    # Remove current version
    if target.exists():
        shutil.rmtree(target)

    # Restore from backup
    shutil.copytree(backup, target)

    # Remove the used backup
    shutil.rmtree(backup)

    print(f"  Rolled back: {skill_name} from {backup.name}")
    return True


# =============================================================================
# Validation
# =============================================================================

def validate_skill_before_deploy(skill_path: str) -> Tuple[bool, str]:
    """
    Run validation script on a skill before deployment.

    Returns:
        Tuple of (passed, message)
    """
    skill_name = get_skill_name(skill_path)
    source = SOURCE_DIR / skill_path

    if not VALIDATE_SCRIPT.exists():
        return True, "Validation script not found, skipping validation"

    try:
        result = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), str(source)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return True, f"Validation passed for {skill_name}"
        else:
            # Extract error summary
            error_lines = [
                line for line in result.stdout.split("\n")
                if "error" in line.lower() or "Error" in line
            ]
            error_summary = "; ".join(error_lines[:3]) if error_lines else "Unknown error"
            return False, f"Validation failed: {error_summary}"

    except subprocess.TimeoutExpired:
        return False, "Validation timed out"
    except Exception as e:
        return False, f"Validation error: {e}"


# =============================================================================
# Deployment Functions
# =============================================================================

def deploy_skill(
    skill_path: str,
    dry_run: bool = False,
    validate: bool = False,
    backup: bool = False,
    version: str = DEFAULT_VERSION,
    manifest: Optional[Dict] = None
) -> Tuple[bool, str]:
    """
    Deploy a single skill to the target directory.

    Returns:
        Tuple of (success, message)
    """
    skill_name = get_skill_name(skill_path)
    source = SOURCE_DIR / skill_path
    target = TARGET_DIR / skill_name

    # Check source exists
    if not source.exists():
        return False, f"Source not found: {source}"

    # Validation
    if validate:
        passed, msg = validate_skill_before_deploy(skill_path)
        if not passed:
            return False, msg

    # Dry run
    if dry_run:
        return True, f"[DRY RUN] Would copy: {source} -> {target}"

    # Backup
    if backup and target.exists():
        backup_path = create_backup(skill_name)
        if backup_path:
            print(f"    Backup created: {backup_path.name}")

    # Remove existing
    if target.exists():
        shutil.rmtree(target)

    # Copy skill directory
    shutil.copytree(source, target)

    # Update manifest
    if manifest is not None:
        checksum = calculate_checksum(target)
        files = get_files_info(target)
        update_manifest(manifest, skill_name, skill_path, version, checksum, files)

    return True, f"Deployed: {skill_name}"


def deploy_lib(dry_run: bool = False, backup: bool = False) -> Tuple[bool, str]:
    """Deploy shared library to skills directory."""
    lib_source = Path(__file__).parent / "lib" / "python"
    lib_target = TARGET_DIR / "_causal_ml_lib"

    if not lib_source.exists():
        return False, f"Lib not found: {lib_source}"

    if dry_run:
        return True, f"[DRY RUN] Would copy lib: {lib_source} -> {lib_target}"

    # Backup
    if backup and lib_target.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"_causal_ml_lib_{timestamp}"
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(lib_target, backup_path)

    if lib_target.exists():
        shutil.rmtree(lib_target)

    shutil.copytree(lib_source, lib_target)
    return True, "Deployed: _causal_ml_lib (shared library)"


def get_skills_by_category(category: str) -> List[str]:
    """Get skill paths for a specific category."""
    if category not in SKILL_CATEGORIES:
        return []
    return SKILL_CATEGORIES[category]


def get_available_categories() -> List[str]:
    """Get list of available categories."""
    return list(SKILL_CATEGORIES.keys())


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deploy causal-ml-skills to Claude Code skills directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Selection options
    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--skill",
        type=str,
        help="Deploy specific skill only"
    )
    selection_group.add_argument(
        "--batch",
        action="store_true",
        help="Deploy all skills without prompts"
    )
    selection_group.add_argument(
        "--category",
        type=str,
        choices=get_available_categories(),
        help="Deploy skills in a specific category"
    )

    # Deployment options
    deploy_group = parser.add_argument_group("Deployment Options")
    deploy_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without deploying"
    )
    deploy_group.add_argument(
        "--validate",
        action="store_true",
        help="Run validation before deploying"
    )
    deploy_group.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before deploying"
    )
    deploy_group.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"Set version for deployment (default: {DEFAULT_VERSION})"
    )
    deploy_group.add_argument(
        "--no-lib",
        action="store_true",
        help="Skip library deployment"
    )

    # Manifest and rollback options
    manifest_group = parser.add_argument_group("Manifest and Rollback Options")
    manifest_group.add_argument(
        "--manifest",
        action="store_true",
        help="Generate/update deployment manifest"
    )
    manifest_group.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous version"
    )
    manifest_group.add_argument(
        "--show-manifest",
        action="store_true",
        help="Show current deployment manifest"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Handle show-manifest
    if args.show_manifest:
        manifest = load_manifest()
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    # Header
    if not args.quiet and not args.json:
        print("=" * 60)
        print("CAUSAL-ML-SKILLS DEPLOYMENT")
        print("=" * 60)
        print(f"Source: {SOURCE_DIR}")
        print(f"Target: {TARGET_DIR}")
        print(f"Version: {args.version}")
        if args.validate:
            print("Validation: ENABLED")
        if args.backup:
            print("Backup: ENABLED")
        if args.manifest:
            print("Manifest: ENABLED")
        print()

    # Ensure target directory exists
    if not TARGET_DIR.exists():
        if not args.quiet:
            print(f"Creating target directory: {TARGET_DIR}")
        if not args.dry_run:
            TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Handle rollback
    if args.rollback:
        if args.skill:
            success = rollback_skill(args.skill, args.dry_run)
            return 0 if success else 1
        else:
            print("ERROR: --rollback requires --skill to specify which skill to rollback")
            return 1

    # Determine skills to deploy
    skills_to_deploy = SKILL_DIRS

    if args.skill:
        skills_to_deploy = [s for s in SKILL_DIRS if get_skill_name(s) == args.skill]
        if not skills_to_deploy:
            print(f"ERROR: Skill not found: {args.skill}")
            print(f"Available skills: {[get_skill_name(s) for s in SKILL_DIRS]}")
            return 1
    elif args.category:
        skills_to_deploy = get_skills_by_category(args.category)
        if not skills_to_deploy:
            print(f"ERROR: No skills found in category: {args.category}")
            return 1
        if not args.quiet:
            print(f"Category: {args.category} ({len(skills_to_deploy)} skills)")
    elif not args.batch:
        # Interactive mode - ask for confirmation
        print(f"About to deploy {len(skills_to_deploy)} skills.")
        print("Skills to deploy:")
        for category, skills in SKILL_CATEGORIES.items():
            print(f"  {category}:")
            for skill in skills:
                print(f"    - {get_skill_name(skill)}")
        print()

        try:
            response = input("Continue? [y/N] ")
            if response.lower() not in ["y", "yes"]:
                print("Deployment cancelled.")
                return 0
        except (KeyboardInterrupt, EOFError):
            print("\nDeployment cancelled.")
            return 0

    # Load manifest if needed
    manifest = load_manifest() if args.manifest else None
    if manifest:
        manifest["version"] = args.version

    # Deploy lib first
    if not args.no_lib:
        if not args.quiet:
            print("Deploying shared library...")
        success, msg = deploy_lib(args.dry_run, args.backup)
        if not args.quiet:
            status = "success" if success else "error"
            prefix = "\033[92m[OK]\033[0m" if success else "\033[91m[FAIL]\033[0m"
            print(f"  {prefix} {msg}")
        print()

    # Initialize progress reporter
    progress = ProgressReporter(len(skills_to_deploy), verbose=not args.quiet and not args.json)

    # Deploy skills
    if not args.quiet and not args.json:
        print(f"Deploying {len(skills_to_deploy)} skills...")

    success_count = 0
    failed_count = 0
    skipped_count = 0
    results = []

    for skill_path in skills_to_deploy:
        skill_name = get_skill_name(skill_path)

        success, msg = deploy_skill(
            skill_path,
            dry_run=args.dry_run,
            validate=args.validate,
            backup=args.backup,
            version=args.version,
            manifest=manifest
        )

        results.append({
            "skill": skill_name,
            "path": skill_path,
            "success": success,
            "message": msg
        })

        if success:
            success_count += 1
            status = "success"
        else:
            if "Validation failed" in msg:
                skipped_count += 1
                status = "skip"
            else:
                failed_count += 1
                status = "error"

        progress.update(msg, status)

    # Save manifest
    if manifest and not args.dry_run:
        save_manifest(manifest)
        if not args.quiet:
            print(f"\nManifest saved to: {MANIFEST_FILE}")

    # Output results
    if args.json:
        output = {
            "version": args.version,
            "timestamp": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "total": len(skills_to_deploy),
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "results": results
        }
        print(json.dumps(output, indent=2))
    else:
        progress.summary(success_count, failed_count, skipped_count)

        if args.dry_run:
            print("\n[DRY RUN] No files were actually copied.")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())
