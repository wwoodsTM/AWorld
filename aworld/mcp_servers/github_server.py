"""
GitHub MCP Server

This module provides a microservice for interacting with the GitHub API through MCP.
It enables retrieving and searching GitHub data including repositories, files, issues,
and user information with proper error handling and rate limit management.

Key features:
- Repository information retrieval and searching
- File content access and directory listing
- Code search across repositories
- User profile and repository information
- Issue tracking and label management

Main functions:
- get_repository: Retrieves detailed information about a GitHub repository
- get_file_content: Fetches and decodes content from repository files
- search_repositories: Finds repositories matching search criteria
- get_user: Retrieves GitHub user profile information
- list_repository_contents: Lists files and directories in a repository
- get_issues: Retrieves issues from a repository with filtering options
"""

import base64
import json
import os
import traceback
from typing import Any, List, Optional

from github import Github, Label
from pydantic import BaseModel, Field

from aworld.logs.util import logger
from aworld.mcp_servers.abc.base import MCPServerBase, mcp
from aworld.mcp_servers.utils import parse_port, run_mcp_server
from aworld.utils import import_package


# Define model classes for different GitHub API responses
class GitHubRepository(BaseModel):
    """Model representing a GitHub repository"""

    id: int
    name: str
    full_name: str
    owner: str
    description: str = ""
    html_url: str
    stars: int
    forks: int
    watchers: int
    language: Optional[str] = None
    default_branch: str
    open_issues_count: int
    created_at: str
    updated_at: str
    topics: List[str] = []
    is_private: bool
    is_fork: bool
    has_wiki: bool
    has_issues: bool


class GitHubFile(BaseModel):
    """Model representing a file in a GitHub repository"""

    name: str
    path: str
    sha: str
    size: int
    url: str
    html_url: str
    type: str  # "file" or "dir"
    content: Optional[str] = None
    encoding: Optional[str] = None


class GitHubUser(BaseModel):
    """Model representing a GitHub user"""

    id: int
    login: str
    name: Optional[str] = None
    avatar_url: str
    html_url: str
    type: str
    public_repos: int
    followers: int
    following: int
    created_at: str
    updated_at: str
    bio: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    blog: Optional[str] = None


class GitHubIssue(BaseModel):
    """Model representing a GitHub issue"""

    id: int
    number: int
    title: str
    body: str = ""
    state: str
    html_url: str
    user: str
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None
    labels: List[str] = []
    assignees: List[str] = []
    comments: int
    is_pull_request: bool = False


class GitHubSearchResult(BaseModel):
    """Model representing search results"""

    query: str
    total_count: int
    items: List[Any]


class GitHubError(BaseModel):
    """Model representing an error in GitHub API processing"""

    error: str
    operation: str


class GitHubServer(MCPServerBase):
    """
    GitHub Server class for interacting with the GitHub API.

    This class provides methods for retrieving repositories, issues, pull requests,
    and user information from GitHub.
    """

    _instance = None
    _github = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(GitHubServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the GitHub server and client"""
        # Import PyGithub package, install if not available
        import_package("github", install_name="PyGithub")
        self._github = self._get_github_instance()
        logger.info("GitHubServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of GitHubServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def handle_error(e: Exception, operation_type: str) -> str:
        """Unified error handling and return standard format error message"""
        error_msg = f"{operation_type} error: {str(e)}"
        logger.error(traceback.format_exc())

        error = GitHubError(error=error_msg, operation=operation_type)

        return error.model_dump_json()

    @staticmethod
    def _get_github_instance() -> Github:
        """
        Create and return a GitHub API instance.

        Returns:
            Authenticated GitHub instance
        """
        token = os.environ.get("GITHUB_ACCESS_TOKEN")
        if token:
            return Github(token)
        else:
            logger.warning(
                "No GitHub token found. Using unauthenticated access with rate limits."
            )
            return Github()

    @mcp
    @classmethod
    def get_repository(
        cls,
        repo_full_name: str = Field(description="Repository full name (owner/repo)"),
    ) -> str:
        """
        Get information about a GitHub repository.

        Args:
            repo_full_name: Repository full name in format "owner/repo"

        Returns:
            JSON string containing repository information
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(repo_full_name, "default") and not isinstance(
                repo_full_name, str
            ):
                repo_full_name = repo_full_name.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Get repository
            repo = instance._github.get_repo(repo_full_name)

            # Create repository object
            repo_obj = GitHubRepository(
                id=repo.id,
                name=repo.name,
                full_name=repo.full_name,
                owner=repo.owner.login,
                description=repo.description or "",
                html_url=repo.html_url,
                stars=repo.stargazers_count,
                forks=repo.forks_count,
                watchers=repo.watchers_count,
                language=repo.language,
                default_branch=repo.default_branch,
                open_issues_count=repo.open_issues_count,
                created_at=repo.created_at.isoformat(),
                updated_at=repo.updated_at.isoformat(),
                topics=repo.get_topics(),
                is_private=repo.private,
                is_fork=repo.fork,
                has_wiki=repo.has_wiki,
                has_issues=repo.has_issues,
            )

            return repo_obj.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Get Repository")

    @mcp
    @classmethod
    def list_repository_contents(
        cls,
        repo_full_name: str = Field(description="Repository full name (owner/repo)"),
        path: str = Field(default="", description="Directory path within repository"),
        ref: str = Field(
            default="", description="Branch, tag, or commit SHA (optional)"
        ),
    ) -> str:
        """
        List contents of a directory in a GitHub repository.

        Args:
            repo_full_name: Repository full name in format "owner/repo"
            path: Directory path within repository (default: root)
            ref: Branch, tag, or commit SHA (optional)

        Returns:
            JSON string containing directory contents
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(repo_full_name, "default") and not isinstance(
                repo_full_name, str
            ):
                repo_full_name = repo_full_name.default

            if hasattr(path, "default") and not isinstance(path, str):
                path = path.default

            if hasattr(ref, "default") and not isinstance(ref, str):
                ref = ref.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Get repository
            repo = instance._github.get_repo(repo_full_name)

            # Get directory contents
            if ref:
                contents = repo.get_contents(path, ref=ref)
            else:
                contents = repo.get_contents(path)

            # Handle file case
            if not isinstance(contents, list):
                raise ValueError(f"Path {path} is a file, not a directory")

            # Process contents
            items = []
            for item in contents:
                file_obj = GitHubFile(
                    name=item.name,
                    path=item.path,
                    sha=item.sha,
                    size=item.size,
                    url=item.url,
                    html_url=item.html_url,
                    type="file" if item.type == "file" else "dir",
                )
                items.append(file_obj)

            # Create result
            result = {
                "repository": repo_full_name,
                "path": path,
                "contents": [item.model_dump() for item in items],
                "count": len(items),
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "List Repository Contents")

    @mcp
    @classmethod
    def search_repositories(
        cls,
        query: str = Field(description="Search query"),
        sort: str = Field(
            default="stars",
            description="Sort method (stars, forks, updated, help-wanted-issues)",
        ),
        order: str = Field(default="desc", description="Sort order (asc, desc)"),
        limit: int = Field(
            default=30, description="Number of results to retrieve (max 100)"
        ),
    ) -> str:
        """
        Search for GitHub repositories based on query.

        Args:
            query: Search query
            sort: Sort method for results
            order: Sort order (asc, desc)
            limit: Number of results to retrieve (max 100)

        Returns:
            JSON string containing search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if hasattr(sort, "default") and not isinstance(sort, str):
                sort = sort.default

            if hasattr(order, "default") and not isinstance(order, str):
                order = order.default

            if hasattr(limit, "default") and not isinstance(limit, int):
                limit = limit.default

            # Validate input
            if limit > 100:
                limit = 100
                logger.warning("Limit capped at 100 results")

            # Get the singleton instance
            instance = cls.get_instance()

            # Search repositories
            repositories = instance._github.search_repositories(
                query=query, sort=sort, order=order
            )

            # Process results
            repos = []
            count = 0
            for repo in repositories:
                if count >= limit:
                    break

                repo_obj = GitHubRepository(
                    id=repo.id,
                    name=repo.name,
                    full_name=repo.full_name,
                    owner=repo.owner.login,
                    description=repo.description or "",
                    html_url=repo.html_url,
                    stars=repo.stargazers_count,
                    forks=repo.forks_count,
                    watchers=repo.watchers_count,
                    language=repo.language,
                    default_branch=repo.default_branch,
                    open_issues_count=repo.open_issues_count,
                    created_at=repo.created_at.isoformat(),
                    updated_at=repo.updated_at.isoformat(),
                    topics=repo.get_topics(),
                    is_private=repo.private,
                    is_fork=repo.fork,
                    has_wiki=repo.has_wiki,
                    has_issues=repo.has_issues,
                )
                repos.append(repo_obj)
                count += 1

            # Create search result
            search_result = GitHubSearchResult(
                query=query,
                total_count=repositories.totalCount,
                items=[repo.model_dump() for repo in repos],
            )

            return search_result.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Search Repositories")

    @mcp
    @classmethod
    def search_code(
        cls,
        query: str = Field(description="Search query"),
        language: str = Field(default="", description="Filter by language (optional)"),
        repo: str = Field(default="", description="Filter by repository (optional)"),
        limit: int = Field(
            default=30, description="Number of results to retrieve (max 100)"
        ),
    ) -> str:
        """
        Search for code in GitHub repositories.

        Args:
            query: Search query
            language: Filter by language (optional)
            repo: Filter by repository in format "owner/repo" (optional)
            limit: Number of results to retrieve (max 100)

        Returns:
            JSON string containing search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if hasattr(repo, "default") and not isinstance(repo, str):
                repo = repo.default

            if hasattr(limit, "default") and not isinstance(limit, int):
                limit = limit.default

            # Validate input
            if limit > 100:
                limit = 100
                logger.warning("Limit capped at 100 results")

            # Build search query
            search_query = query
            if language:
                search_query += f" language:{language}"
            if repo:
                search_query += f" repo:{repo}"

            # Get the singleton instance
            instance = cls.get_instance()

            # Search code
            code_results = instance._github.search_code(query=search_query)

            # Process results
            code_items = []
            count = 0
            for code in code_results:
                if count >= limit:
                    break

                code_item = {
                    "name": code.name,
                    "path": code.path,
                    "sha": code.sha,
                    "url": code.url,
                    "html_url": code.html_url,
                    "repository": code.repository.full_name,
                    "score": code.score,
                }
                code_items.append(code_item)
                count += 1

            # Create search result
            result = {
                "query": search_query,
                "total_count": code_results.totalCount,
                "items": code_items,
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "Search Code")

    @mcp
    @classmethod
    def get_user(
        cls,
        username: str = Field(description="GitHub username"),
    ) -> str:
        """
        Get information about a GitHub user.

        Args:
            username: GitHub username

        Returns:
            JSON string containing user information
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(username, "default") and not isinstance(username, str):
                username = username.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Get user
            user = instance._github.get_user(username)

            # Create user object
            user_obj = GitHubUser(
                id=user.id,
                login=user.login,
                name=user.name,
                avatar_url=user.avatar_url,
                html_url=user.html_url,
                type=user.type,
                public_repos=user.public_repos,
                followers=user.followers,
                following=user.following,
                created_at=user.created_at.isoformat(),
                updated_at=user.updated_at.isoformat(),
                bio=user.bio,
                company=user.company,
                location=user.location,
                email=user.email,
                blog=user.blog,
            )

            return user_obj.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Get User")

    @mcp
    @classmethod
    def get_user_repositories(
        cls,
        username: str = Field(description="GitHub username"),
        sort: str = Field(
            default="updated",
            description="Sort method (created, updated, pushed, full_name)",
        ),
        direction: str = Field(
            default="desc", description="Sort direction (asc, desc)"
        ),
        limit: int = Field(
            default=30, description="Number of repositories to retrieve (max 100)"
        ),
    ) -> str:
        """
        Get repositories owned by a GitHub user.

        Args:
            username: GitHub username
            sort: Sort method for repositories
            direction: Sort direction (asc, desc)
            limit: Number of repositories to retrieve (max 100)

        Returns:
            JSON string containing user repositories
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(username, "default") and not isinstance(username, str):
                username = username.default

            if hasattr(sort, "default") and not isinstance(sort, str):
                sort = sort.default

            if hasattr(direction, "default") and not isinstance(direction, str):
                direction = direction.default

            if hasattr(limit, "default") and not isinstance(limit, int):
                limit = limit.default

            # Validate input
            if limit > 100:
                limit = 100
                logger.warning("Limit capped at 100 repositories")

            # Get the singleton instance
            instance = cls.get_instance()

            # Get user
            user = instance._github.get_user(username)

            # Get user repositories
            repositories = user.get_repos(sort=sort, direction=direction)

            # Process repositories
            repos = []
            count = 0
            for repo in repositories:
                if count >= limit:
                    break

                repo_obj = GitHubRepository(
                    id=repo.id,
                    name=repo.name,
                    full_name=repo.full_name,
                    owner=repo.owner.login,
                    description=repo.description or "",
                    html_url=repo.html_url,
                    stars=repo.stargazers_count,
                    forks=repo.forks_count,
                    watchers=repo.watchers_count,
                    language=repo.language,
                    default_branch=repo.default_branch,
                    open_issues_count=repo.open_issues_count,
                    created_at=repo.created_at.isoformat(),
                    updated_at=repo.updated_at.isoformat(),
                    topics=repo.get_topics(),
                    is_private=repo.private,
                    is_fork=repo.fork,
                    has_wiki=repo.has_wiki,
                    has_issues=repo.has_issues,
                )
                repos.append(repo_obj)
                count += 1

            # Create result
            result = {
                "username": username,
                "repositories": [repo.model_dump() for repo in repos],
                "count": len(repos),
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "Get User Repositories")

    @mcp
    @classmethod
    def get_issues(
        cls,
        repo_full_name: str = Field(description="Repository full name (owner/repo)"),
        state: str = Field(default="open", description="Issue state (open, closed)"),
        filter_labels: List[Label] = Field(
            default=[],
            description="Filter issues by labels, apply `and` logic for multiple labels",
        ),
        sort: str = Field(
            default="created", description="Sort method (created, updated, comments)"
        ),
        direction: str = Field(
            default="desc", description="Sort direction (asc, desc)"
        ),
        limit: int = Field(
            default=375, description="Number of issues to retrieve (max 1024)"
        ),
    ) -> str:
        """
        Get issues from a GitHub repository.

        Args:
            repo_full_name: Repository full name in format "owner/repo"
            state: Issue state (open, closed)
            sort: Sort method for issues
            direction: Sort direction (asc, desc)
            limit: Number of issues to retrieve (max 1024)
            filter_labels: Filter issues by labels, always validate labels existing before passing

        Returns:
            JSON string containing repository issues
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(repo_full_name, "default") and not isinstance(
                repo_full_name, str
            ):
                repo_full_name = repo_full_name.default

            if hasattr(state, "default") and not isinstance(state, str):
                state = state.default

            if hasattr(filter_labels, "default") and not isinstance(
                filter_labels, list
            ):
                filter_labels = filter_labels.default

            if hasattr(sort, "default") and not isinstance(sort, str):
                sort = sort.default

            if hasattr(direction, "default") and not isinstance(direction, str):
                direction = direction.default

            if hasattr(limit, "default") and not isinstance(limit, int):
                limit = limit.default

            # Validate input
            if limit > 1024:
                limit = 1024
                logger.warning("Limit capped at 1024 issues")

            # Get the singleton instance
            instance = cls.get_instance()

            # Get repository
            repo = instance._github.get_repo(repo_full_name)

            # Get issues
            issues = repo.get_issues(
                state=state, sort=sort, direction=direction, labels=filter_labels
            )

            # Process issues
            issue_list = []
            count = 0

            for issue in issues:
                if count >= limit:
                    break

                # Skip pull requests if they're returned with issues
                is_pull_request = issue.pull_request is not None

                issue_obj = GitHubIssue(
                    id=issue.id,
                    number=issue.number,
                    title=issue.title,
                    body=issue.body or "",
                    state=issue.state,
                    html_url=issue.html_url,
                    user=issue.user.login,
                    created_at=issue.created_at.isoformat(),
                    updated_at=issue.updated_at.isoformat(),
                    closed_at=issue.closed_at.isoformat() if issue.closed_at else None,
                    labels=[label.name for label in issue.labels],
                    assignees=[assignee.login for assignee in issue.assignees],
                    comments=issue.comments,
                    is_pull_request=is_pull_request,
                )
                issue_list.append(issue_obj)
                count += 1

            # Create result
            result = {
                "repository": repo_full_name,
                "issues": [issue.model_dump() for issue in issue_list],
                "count": len(issue_list),
                "filter_applied": len(filter_labels) > 0,
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "Get Issues")

    @mcp
    @classmethod
    def get_labels(
        cls,
        repo_full_name: str = Field(description="Repository full name (owner/repo)"),
    ) -> str:
        """
        Get all labels from a GitHub repository.

        Args:
            repo_full_name: Repository full name in format "owner/repo"

        Returns:
            JSON string containing repository labels
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(repo_full_name, "default") and not isinstance(
                repo_full_name, str
            ):
                repo_full_name = repo_full_name.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Get repository
            repo = instance._github.get_repo(repo_full_name)

            # Get labels
            labels = repo.get_labels()

            # Process labels
            label_list = []
            for label in labels:
                label_obj = {
                    "name": label.name,
                    "description": label.description or "",
                    "color": label.color,
                    "url": label.url,
                }
                label_list.append(label_obj)

            # Create result
            result = {
                "repository": repo_full_name,
                "labels": label_list,
                "count": len(label_list),
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "Get Labels")

    @mcp
    @classmethod
    def get_file_content(
        cls,
        repo_full_name: str = Field(description="Repository full name (owner/repo)"),
        file_path: str = Field(description="Path to the file within the repository"),
        ref: str = Field(
            default="", description="Branch, tag, or commit SHA (optional)"
        ),
    ) -> str:
        """
        Get content of a file from a GitHub repository.

        Args:
            repo_full_name: Repository full name in format "owner/repo"
            file_path: Path to the file within the repository
            ref: Branch, tag, or commit SHA (optional)

        Returns:
            JSON string containing file content and metadata
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(repo_full_name, "default") and not isinstance(
                repo_full_name, str
            ):
                repo_full_name = repo_full_name.default

            if hasattr(file_path, "default") and not isinstance(file_path, str):
                file_path = file_path.default

            if hasattr(ref, "default") and not isinstance(ref, str):
                ref = ref.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Get repository
            repo = instance._github.get_repo(repo_full_name)

            # Get file content
            if ref:
                content_file = repo.get_contents(file_path, ref=ref)
            else:
                content_file = repo.get_contents(file_path)

            # Handle directory case
            if isinstance(content_file, list):
                raise ValueError(f"Path {file_path} is a directory, not a file")

            # Decode content if it's base64 encoded
            if content_file.encoding == "base64":
                decoded_content = base64.b64decode(content_file.content).decode("utf-8")
            else:
                decoded_content = content_file.content

            # Create file object
            file_obj = GitHubFile(
                name=content_file.name,
                path=content_file.path,
                sha=content_file.sha,
                size=content_file.size,
                url=content_file.url,
                html_url=content_file.html_url,
                type="file",
                content=decoded_content,
                encoding="utf-8",
            )

            return file_obj.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Get File Content")


# Main function
if __name__ == "__main__":
    port = parse_port()

    github_server = GitHubServer.get_instance()
    logger.info("GitHubServer initialized and ready to handle requests")

    run_mcp_server(
        "GitHub Server",
        funcs=[
            github_server.get_repository,
            github_server.list_repository_contents,
            github_server.search_repositories,
            github_server.search_code,
            github_server.get_user,
            github_server.get_user_repositories,
            github_server.get_labels,
            github_server.get_issues,
            github_server.get_file_content,
        ],
        port=port,
    )
