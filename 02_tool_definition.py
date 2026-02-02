"""
Tool Calling - Part 1: Tool Definition Methods
@tool decorator vs Pydantic schema
"""

from typing import Annotated
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field


# =============================================================================
# Method 1: @tool decorator (Simple)
# =============================================================================

@tool
def get_weather_simple(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"


# =============================================================================
# Method 2: @tool with type hints (Better docs)
# =============================================================================

@tool
def get_weather_typed(
    city: Annotated[str, "The city name to get weather for"],
    unit: Annotated[str, "Temperature unit: celsius or fahrenheit"] = "celsius"
) -> str:
    """Get current weather for a city with specified unit."""
    temp = "22°C" if unit == "celsius" else "72°F"
    return f"Weather in {city}: Sunny, {temp}"


# =============================================================================
# Method 3: Pydantic schema (Full control)
# =============================================================================

class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(description="The city name to get weather for")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")
    include_forecast: bool = Field(default=False, description="Include 3-day forecast")


@tool(args_schema=WeatherInput)
def get_weather_pydantic(city: str, unit: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather for a city with full options."""
    temp = "22°C" if unit == "celsius" else "72°F"
    result = f"Weather in {city}: Sunny, {temp}"
    if include_forecast:
        result += "\nForecast: Sun, Mon, Tue - Sunny"
    return result


# =============================================================================
# Method 4: StructuredTool (Programmatic)
# =============================================================================

def _search_impl(query: str, max_results: int = 5) -> str:
    """Implementation function."""
    return f"Search results for '{query}': Found {max_results} results"


class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results to return")


search_tool = StructuredTool.from_function(
    func=_search_impl,
    name="web_search",
    description="Search the web for information",
    args_schema=SearchInput,
)


# =============================================================================
# Compare tool schemas
# =============================================================================

if __name__ == "__main__":
    tools = [get_weather_simple, get_weather_typed, get_weather_pydantic, search_tool]

    for t in tools:
        print(f"\n{'='*60}")
        print(f"Tool: {t.name}")
        print(f"Description: {t.description}")
        print(f"Args Schema: {t.args_schema.model_json_schema() if t.args_schema else 'None'}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
| Method          | Pros                          | Cons                    |
|-----------------|-------------------------------|-------------------------|
| @tool simple    | Minimal code                  | No arg descriptions     |
| @tool typed     | Annotated descriptions        | Verbose for many args   |
| @tool + Pydantic| Full control, validation      | More boilerplate        |
| StructuredTool  | Programmatic, reusable        | Most verbose            |

Recommendation:
- Simple tools: @tool decorator
- Production tools: @tool + Pydantic schema (validation + docs)
""")
