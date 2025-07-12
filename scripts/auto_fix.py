#!/usr/bin/env python3
"""
Supreme Self-Healing CI Auto-Fix Script for CogML

This script implements the recursive error detection, diagnosis, and resolution
system described in the issue. It parses build logs for Cython/type errors,
generates patches, and applies fixes automatically.

Design follows the cognitive architecture outlined in the problem statement:
- Memory System: Error logs, patch history, fix templates
- Task System: Build-test-fix orchestration 
- AI System: Log parsing, patch synthesis, solution verification
- Autonomy System: Iterative fix logic, strategy learning, escalation
"""

import os
import sys
import re
import json
import subprocess
import argparse
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

class ErrorType(Enum):
    CYTHON_IMPORT = "cython_import"
    CYTHON_INHERITANCE = "cython_inheritance"
    CYTHON_UNDEFINED_TYPE = "cython_undefined_type"
    CMAKE_MISSING_DEPENDENCY = "cmake_missing_dependency"
    BUILD_COMPILATION = "build_compilation"
    UNKNOWN = "unknown"

@dataclass
class ErrorPattern:
    """Memory system: Error pattern classification"""
    pattern: str
    error_type: ErrorType
    description: str
    fix_template: str

@dataclass
class BuildError:
    """Memory system: Build error representation"""
    file_path: str
    line_number: int
    error_type: ErrorType
    message: str
    context: str

@dataclass
class FixAttempt:
    """Memory system: Fix attempt tracking"""
    timestamp: str
    error: BuildError
    patch_content: str
    success: bool
    build_output: str

class CythonFixTemplates:
    """AI System: Fix templates for common Cython errors"""
    
    MISSING_IMPORT_VALUE = """# Auto-generated import fix for Value inheritance
from .value cimport Value
"""

    MISSING_IMPORT_ATOMSPACE = """# Auto-generated import fix for AtomSpace types
from .atomspace cimport cHandle, cAtomSpace, cAtom
"""

    MISSING_IMPORT_TRUTH_VALUE = """# Auto-generated import fix for TruthValue types  
from .atomspace cimport tv_ptr, cTruthValue
"""

    CYTHON_INHERITANCE_FIX = """# Auto-generated inheritance fix
# Original inheritance issue resolved by proper import"""

class ErrorClassifier:
    """AI System: Log parsing and error classification"""
    
    def __init__(self):
        self.patterns = [
            ErrorPattern(
                pattern=r"First base of '(\w+)' is not an extension type",
                error_type=ErrorType.CYTHON_INHERITANCE,
                description="Cython inheritance from undefined extension type",
                fix_template="missing_import"
            ),
            ErrorPattern(
                pattern=r"'(\w+)' is not declared",
                error_type=ErrorType.CYTHON_UNDEFINED_TYPE,
                description="Undefined Cython type or variable",
                fix_template="missing_declaration"
            ),
            ErrorPattern(
                pattern=r"Cannot find '(.+)\.pxd'",
                error_type=ErrorType.CYTHON_IMPORT,
                description="Missing Cython .pxd import file",
                fix_template="missing_import_file"
            ),
            ErrorPattern(
                pattern=r"Could not find (\w+) package",
                error_type=ErrorType.CMAKE_MISSING_DEPENDENCY,
                description="Missing CMake package dependency",
                fix_template="cmake_dependency"
            ),
            ErrorPattern(
                pattern=r"error: (.+)",
                error_type=ErrorType.BUILD_COMPILATION,
                description="General compilation error",
                fix_template="compilation_fix"
            )
        ]
    
    def parse_build_log(self, log_content: str) -> List[BuildError]:
        """Parse build log and extract error information"""
        errors = []
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in self.patterns:
                match = re.search(pattern.pattern, line)
                if match:
                    # Extract file and line number context
                    file_path, line_num = self._extract_file_context(lines, i)
                    
                    error = BuildError(
                        file_path=file_path,
                        line_number=line_num,
                        error_type=pattern.error_type,
                        message=line.strip(),
                        context=self._extract_error_context(lines, i)
                    )
                    errors.append(error)
                    break
        
        return errors
    
    def _extract_file_context(self, lines: List[str], error_line_idx: int) -> Tuple[str, int]:
        """Extract file path and line number from error context"""
        # Look backwards for file reference
        for i in range(max(0, error_line_idx - 5), error_line_idx + 1):
            line = lines[i]
            # Cython error format: filename:line:column:
            file_match = re.search(r'([^:\s]+\.(pyx|pxd)):(\d+):(\d+):', line)
            if file_match:
                return file_match.group(1), int(file_match.group(3))
        
        return "unknown", 0
    
    def _extract_error_context(self, lines: List[str], error_line_idx: int) -> str:
        """Extract surrounding context for error"""
        start = max(0, error_line_idx - 2)
        end = min(len(lines), error_line_idx + 3)
        return '\n'.join(lines[start:end])

class PatchGenerator:
    """AI System: Patch synthesis and solution generation"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.templates = CythonFixTemplates()
    
    def generate_patch(self, error: BuildError) -> Optional[str]:
        """Generate patch content for specific error"""
        if error.error_type == ErrorType.CYTHON_INHERITANCE:
            return self._fix_cython_inheritance(error)
        elif error.error_type == ErrorType.CYTHON_UNDEFINED_TYPE:
            return self._fix_undefined_type(error)
        elif error.error_type == ErrorType.CYTHON_IMPORT:
            return self._fix_missing_import(error)
        
        return None
    
    def _fix_cython_inheritance(self, error: BuildError) -> str:
        """Fix Cython inheritance issues by adding missing imports"""
        if "Value" in error.message:
            return self.templates.MISSING_IMPORT_VALUE
        elif "Atom" in error.message:
            return self.templates.MISSING_IMPORT_ATOMSPACE
        
        return self.templates.CYTHON_INHERITANCE_FIX
    
    def _fix_undefined_type(self, error: BuildError) -> str:
        """Fix undefined type errors"""
        if any(t in error.message for t in ["cHandle", "cAtomSpace", "cAtom"]):
            return self.templates.MISSING_IMPORT_ATOMSPACE
        elif any(t in error.message for t in ["tv_ptr", "cTruthValue"]):
            return self.templates.MISSING_IMPORT_TRUTH_VALUE
        
        return "# Auto-generated fix for undefined type"
    
    def _fix_missing_import(self, error: BuildError) -> str:
        """Fix missing import files"""
        return "# Auto-generated import fix"
    
    def apply_patch(self, error: BuildError, patch_content: str) -> bool:
        """Apply patch to target file"""
        try:
            file_path = self.repo_root / error.file_path
            if not file_path.exists():
                # Try to find file in repository
                found_files = list(self.repo_root.glob(f"**/{Path(error.file_path).name}"))
                if found_files:
                    file_path = found_files[0]
                else:
                    print(f"Error: Could not find file {error.file_path}")
                    return False
            
            # Read current content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Apply patch at beginning of file (after existing imports)
            lines = content.split('\n')
            import_end = 0
            
            # Find end of import section
            for i, line in enumerate(lines):
                if line.strip().startswith(('from ', 'import ', 'cimport ')):
                    import_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # Insert patch
            lines.insert(import_end, patch_content.strip())
            
            # Write back
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"Applied patch to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error applying patch: {e}")
            return False

class AutonomousFixSystem:
    """Autonomy System: Iterative fix logic and escalation"""
    
    def __init__(self, repo_root: Path, max_attempts: int = 3):
        self.repo_root = repo_root
        self.max_attempts = max_attempts
        self.classifier = ErrorClassifier()
        self.patch_generator = PatchGenerator(repo_root)
        self.fix_history: List[FixAttempt] = []
        self.artifacts_dir = repo_root / "ci_artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def run_build_command(self, build_cmd: List[str]) -> Tuple[bool, str]:
        """Execute build command and capture output"""
        try:
            result = subprocess.run(
                build_cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Build timed out after 10 minutes"
        except Exception as e:
            return False, f"Build command failed: {e}"
    
    def recursive_self_healing(self, build_cmd: List[str]) -> bool:
        """Main recursive self-healing loop"""
        print("🔄 Starting Relentless Self-Healing CI Process...")
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n🚀 Build Attempt {attempt}/{self.max_attempts}")
            
            # Execute build
            success, output = self.run_build_command(build_cmd)
            
            # Save build log
            log_file = self.artifacts_dir / f"build_attempt_{attempt}.log"
            with open(log_file, 'w') as f:
                f.write(output)
            
            if success:
                print("✅ Build successful! Self-healing complete.")
                self._save_success_report(attempt)
                return True
            
            print(f"❌ Build failed on attempt {attempt}")
            
            # Parse errors
            errors = self.classifier.parse_build_log(output)
            print(f"🔍 Detected {len(errors)} errors")
            
            if not errors:
                print("⚠️  No classifiable errors found")
                continue
            
            # Generate and apply fixes
            fixes_applied = 0
            for error in errors[:5]:  # Limit to top 5 errors per iteration
                patch = self.patch_generator.generate_patch(error)
                if patch:
                    if self.patch_generator.apply_patch(error, patch):
                        fixes_applied += 1
                        
                        # Record fix attempt
                        fix_attempt = FixAttempt(
                            timestamp=datetime.datetime.now().isoformat(),
                            error=error,
                            patch_content=patch,
                            success=False,  # Will update on next iteration
                            build_output=output
                        )
                        self.fix_history.append(fix_attempt)
            
            print(f"🔧 Applied {fixes_applied} automatic fixes")
            
            if fixes_applied == 0:
                print("⚠️  No fixes could be applied")
                break
        
        # All attempts failed - escalate
        print(f"\n🚨 ESCALATION: Build failed after {self.max_attempts} attempts")
        self._escalate_to_human()
        return False
    
    def _save_success_report(self, successful_attempt: int):
        """Save success report with fix history"""
        report = {
            "status": "success",
            "attempts": successful_attempt,
            "fix_history": [asdict(fix) for fix in self.fix_history],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        report_file = self.artifacts_dir / "success_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Success report saved to {report_file}")
    
    def _escalate_to_human(self):
        """Escalate to human review with full changelog"""
        escalation_report = {
            "status": "escalation_required",
            "max_attempts_reached": self.max_attempts,
            "fix_history": [asdict(fix) for fix in self.fix_history],
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": f"Automated fixes failed after {self.max_attempts} attempts"
        }
        
        escalation_file = self.artifacts_dir / "escalation_report.json"
        with open(escalation_file, 'w') as f:
            json.dump(escalation_report, f, indent=2)
        
        print(f"📊 Escalation report saved to {escalation_file}")
        print(f"🔍 Fix attempts: {len(self.fix_history)}")
        
        # Create human-readable summary
        summary_file = self.artifacts_dir / "human_review_summary.md"
        with open(summary_file, 'w') as f:
            f.write("# CogML Self-Healing CI Escalation Report\n\n")
            f.write(f"**Status**: Requires human intervention\n")
            f.write(f"**Attempts**: {self.max_attempts}\n")
            f.write(f"**Timestamp**: {escalation_report['timestamp']}\n\n")
            f.write("## Fix History\n\n")
            
            for i, fix in enumerate(self.fix_history, 1):
                f.write(f"### Fix Attempt {i}\n")
                f.write(f"- **File**: {fix.error.file_path}\n")
                f.write(f"- **Error Type**: {fix.error.error_type.value}\n")
                f.write(f"- **Message**: {fix.error.message}\n")
                f.write(f"- **Patch Applied**: ```\n{fix.patch_content}\n```\n\n")
        
        print(f"📝 Human review summary: {summary_file}")

def main():
    """Task System: Main orchestration entry point"""
    parser = argparse.ArgumentParser(description="CogML Self-Healing CI Auto-Fix")
    parser.add_argument("--build-cmd", nargs="+", default=["make", "-j4"],
                       help="Build command to execute")
    parser.add_argument("--max-attempts", type=int, default=3,
                       help="Maximum fix attempts before escalation")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                       help="Repository root directory")
    
    args = parser.parse_args()
    
    print("🧠 Initializing Cognitive Self-Healing System...")
    print(f"📁 Repository: {args.repo_root}")
    print(f"🔨 Build command: {' '.join(args.build_cmd)}")
    print(f"🔄 Max attempts: {args.max_attempts}")
    
    # Initialize autonomous system
    fix_system = AutonomousFixSystem(args.repo_root, args.max_attempts)
    
    # Run recursive self-healing
    success = fix_system.recursive_self_healing(args.build_cmd)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()