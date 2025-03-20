# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import importlib
import subprocess
from typing import Optional
from importlib import metadata
from aworld.logs.util import logger


def import_package(
        package_name: str,
        install_name: Optional[str] = None,
        version: Optional[str] = None,
        installer: str = 'pip',
        timeout: int = 300
) -> object:
    """
    Import and install package if not available.

    Args:
        package_name: Name of the package to import
        install_name: Name of the package to install (if different from import name)
        version: Required version of the package
        installer: Package installer to use ('pip' or 'conda')
        timeout: Installation timeout in seconds

    Returns:
        Imported module

    Raises:
        ImportError: If package cannot be imported or installed
        TimeoutError: If installation exceeds timeout
    """
    try:
        module = importlib.import_module(package_name)

        # Check version if specified
        if version:
            try:
                installed_version = metadata.version(package_name)
                if installed_version != version:
                    logger.warning(
                        f"Package {package_name} version mismatch. "
                        f"Required: {version}, Installed: {installed_version}"
                    )
            except metadata.PackageNotFoundError:
                logger.warning(f"Could not determine version for {package_name}")

        return module

    except ImportError:
        if install_name is None:
            install_name = package_name

        logger.warning(f"Missing {package_name} package. Attempting to install {install_name}")

        try:
            cmd = _get_install_command(installer, install_name, version)
            _execute_install_command(cmd, timeout)

            # Try importing again after installation
            return importlib.import_module(package_name)

        except Exception as e:
            logger.error(f"Failed to install {install_name}: {str(e)}")
            raise


def _get_install_command(installer: str, package_name: str, version: Optional[str] = None) -> list:
    """
    Generate installation command based on specified installer.
    """
    if installer == 'pip':
        cmd = ['pip', 'install']
        if version:
            cmd.append(f'{package_name}=={version}')
        else:
            cmd.append(package_name)
    elif installer == 'conda':
        cmd = ['conda', 'install', '-y', package_name]
        if version:
            cmd.extend([f'={version}'])
    else:
        raise ValueError(f"Unsupported installer: {installer}")

    return cmd


def _execute_install_command(cmd: list, timeout: int) -> None:
    """
    Execute package installation command.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise TimeoutError(f"Package installation timed out after {timeout} seconds")

    if process.returncode != 0:
        raise ImportError(f"Installation failed: {stderr.decode()}")
