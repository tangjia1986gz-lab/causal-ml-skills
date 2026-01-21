"""
Deploy causal-ml-skills to Claude Code skills directory.

Usage:
    python deploy.py           # Deploy all skills
    python deploy.py --dry-run # Preview deployment
    python deploy.py --skill estimator-did  # Deploy specific skill
"""

import shutil
import argparse
from pathlib import Path


# Configuration
SOURCE_DIR = Path(__file__).parent / "skills"
TARGET_DIR = Path(r"C:\Users\tangj\.claude\skills")

# Skill directories to deploy
SKILL_DIRS = [
    "infrastructure/setup-causal-ml-env",
    "classic-methods/causal-concept-guide",
    "classic-methods/estimator-did",
    "classic-methods/estimator-rd",
    "classic-methods/estimator-iv",
    "classic-methods/estimator-psm",
    "ml-foundation/ml-preprocessing",
    "ml-foundation/ml-model-linear",
    "ml-foundation/ml-model-tree",
    "ml-foundation/ml-model-advanced",
    "causal-ml/causal-ddml",
    "causal-ml/causal-forest",
    "causal-ml/causal-mediation-ml",
    "causal-ml/paper-replication-workflow",
]


def get_skill_name(skill_path: str) -> str:
    """Extract skill name from path."""
    return skill_path.split("/")[-1]


def deploy_skill(skill_path: str, dry_run: bool = False) -> bool:
    """Deploy a single skill to the target directory."""
    skill_name = get_skill_name(skill_path)
    source = SOURCE_DIR / skill_path
    target = TARGET_DIR / skill_name

    if not source.exists():
        print(f"  ERROR: Source not found: {source}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would copy: {source} -> {target}")
        return True

    # Remove existing
    if target.exists():
        shutil.rmtree(target)

    # Copy skill directory
    shutil.copytree(source, target)
    print(f"  Deployed: {skill_name}")
    return True


def deploy_lib(dry_run: bool = False) -> bool:
    """Deploy shared library to skills directory."""
    lib_source = Path(__file__).parent / "lib" / "python"
    lib_target = TARGET_DIR / "_causal_ml_lib"

    if not lib_source.exists():
        print(f"  ERROR: Lib not found: {lib_source}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would copy lib: {lib_source} -> {lib_target}")
        return True

    if lib_target.exists():
        shutil.rmtree(lib_target)

    shutil.copytree(lib_source, lib_target)
    print(f"  Deployed: _causal_ml_lib (shared library)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy causal-ml-skills")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deploying")
    parser.add_argument("--skill", type=str, help="Deploy specific skill only")
    parser.add_argument("--no-lib", action="store_true", help="Skip library deployment")
    args = parser.parse_args()

    print("=" * 60)
    print("CAUSAL-ML-SKILLS DEPLOYMENT")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print()

    if not TARGET_DIR.exists():
        print(f"Creating target directory: {TARGET_DIR}")
        if not args.dry_run:
            TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Deploy lib first
    if not args.no_lib:
        print("Deploying shared library...")
        deploy_lib(args.dry_run)
        print()

    # Deploy skills
    skills_to_deploy = SKILL_DIRS
    if args.skill:
        skills_to_deploy = [s for s in SKILL_DIRS if get_skill_name(s) == args.skill]
        if not skills_to_deploy:
            print(f"ERROR: Skill not found: {args.skill}")
            print(f"Available skills: {[get_skill_name(s) for s in SKILL_DIRS]}")
            return 1

    print(f"Deploying {len(skills_to_deploy)} skills...")
    success = 0
    for skill_path in skills_to_deploy:
        if deploy_skill(skill_path, args.dry_run):
            success += 1

    print()
    print("=" * 60)
    print(f"Deployment complete: {success}/{len(skills_to_deploy)} skills deployed")

    if args.dry_run:
        print("\n[DRY RUN] No files were actually copied.")

    return 0 if success == len(skills_to_deploy) else 1


if __name__ == "__main__":
    exit(main())
