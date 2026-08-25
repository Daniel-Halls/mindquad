"""BIDS dataset initializer script.

This module provides classes to create the root BIDS directory structure
and required metadata files such as dataset_description.json, README, and
.bidsignore.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class BIDSMetadata:
    """Class representing dataset metadata for dataset_description.json."""

    def __init__(
        self,
        name: str = "Multimodal Neuroimaging Study",
        bids_version: str = "1.9.0",
        dataset_type: str = "raw",
        license_str: str = "CC0",
        authors: Optional[List[str]] = None,
    ) -> None:
        """Initialize BIDSMetadata.

        Args:
            name: Human-readable name of the dataset.
            bids_version: BIDS specification version string.
            dataset_type: Type of dataset ('raw' or 'derivative').
            license_str: License specification string.
            authors: List of author names.
        """
        self._name = name
        self._bids_version = bids_version
        self._dataset_type = dataset_type
        self._license = license_str
        self._authors = authors or ["Study Investigators"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation.

        Returns:
            Dictionary formatted for BIDS dataset_description.json.
        """
        return {
            "Name": self._name,
            "BIDSVersion": self._bids_version,
            "DatasetType": self._dataset_type,
            "License": self._license,
            "Authors": self._authors,
        }


class BIDSInitializer:
    """Class to initialize root BIDS directory and metadata files."""

    def __init__(self) -> None:
        """Initialize BIDSInitializer."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def _write_dataset_description(
        self, bids_dir: Path, metadata: BIDSMetadata
    ) -> Path:
        """Write dataset_description.json to BIDS root.

        Args:
            bids_dir: Root BIDS directory.
            metadata: BIDSMetadata instance.

        Returns:
            Path to written dataset_description.json.
        """
        desc_path = bids_dir / "dataset_description.json"
        with open(desc_path, "w", encoding="utf-8") as file_handle:
            json.dump(metadata.to_dict(), file_handle, indent=2)
        self._logger.info("Created %s", desc_path)
        return desc_path

    def _write_readme(self, bids_dir: Path, name: str) -> Path:
        """Write default README file if not already present.

        Args:
            bids_dir: Root BIDS directory.
            name: Dataset name.

        Returns:
            Path to written README file.
        """
        readme_path = bids_dir / "README"
        if not readme_path.exists():
            readme_content = (
                f"# {name}\n\nBIDS-compliant neuroimaging dataset.\n"
            )
            readme_path.write_text(readme_content, encoding="utf-8")
            self._logger.info("Created %s", readme_path)
        return readme_path

    def _write_bidsignore(self, bids_dir: Path) -> Path:
        """Write default .bidsignore file.

        Args:
            bids_dir: Root BIDS directory.

        Returns:
            Path to written .bidsignore file.
        """
        ignore_path = bids_dir / ".bidsignore"
        if not ignore_path.exists():
            ignore_content = "*.tmp\n*.log\n.DS_Store\nwork/\n"
            ignore_path.write_text(ignore_content, encoding="utf-8")
            self._logger.info("Created %s", ignore_path)
        return ignore_path

    def initialize_bids_root(
        self,
        bids_dir: Path,
        metadata: Optional[BIDSMetadata] = None,
    ) -> Path:
        """Initialize root BIDS directory with all standard files.

        Args:
            bids_dir: Root BIDS directory.
            metadata: Optional BIDSMetadata instance.

        Returns:
            Path to dataset_description.json.
        """
        bids_dir.mkdir(parents=True, exist_ok=True)
        meta = metadata or BIDSMetadata()
        desc_path = self._write_dataset_description(bids_dir, meta)
        self._write_readme(bids_dir, meta.to_dict().get("Name", "Study"))
        self._write_bidsignore(bids_dir)
        return desc_path


class BIDSInitializerApp:
    """CLI runner for BIDS initialization."""

    def __init__(self) -> None:
        """Initialize BIDSInitializerApp."""
        self._initializer = BIDSInitializer()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI argument parser.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Initialize BIDS directory structure and metadata."
        )
        parser.add_argument(
            "--bids-dir",
            type=Path,
            required=True,
            help="Path to BIDS root directory.",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="tES-FUS Multimodal Neuroimaging Study",
            help="Dataset name.",
        )
        parser.add_argument(
            "--bids-version",
            type=str,
            default="1.9.0",
            help="BIDS version.",
        )
        parser.add_argument(
            "--license",
            type=str,
            default="CC0",
            help="Dataset license.",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Execute the CLI application.

        Args:
            args: Optional command line arguments.

        Returns:
            Exit code integer (0 for success, non-zero for error).
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        metadata = BIDSMetadata(
            name=parsed_args.name,
            bids_version=parsed_args.bids_version,
            license_str=parsed_args.license,
        )

        self._initializer.initialize_bids_root(
            parsed_args.bids_dir,
            metadata,
        )
        return 0


def main() -> None:
    """Main execution function for CLI script."""
    app = BIDSInitializerApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
