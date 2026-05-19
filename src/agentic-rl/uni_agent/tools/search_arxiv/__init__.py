"""arXiv search tool definition."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Search recent arXiv papers for a topic and return candidate papers with metadata and abstracts.
Use this tool when you want the model to read abstracts and produce a most relevant paper list.
""".strip()


class SearchArxivArguments(BaseModel):
    query: str = Field(description="Topic or keyword query for arXiv paper search.")
    max_results: int = Field(default=8, description="Maximum number of recent candidate papers to return.")
    days: int = Field(default=180, description="Only keep papers updated within the last N days.")


@register_tool("search_arxiv")
class SearchArxivTool(AbstractTool):
    @property
    def name(self) -> str:
        return "search_arxiv"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "search_arxiv"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=SearchArxivArguments,
        )

    def get_install_command(self) -> str | None:
        return None
