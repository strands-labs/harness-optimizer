# appworld_strands_tool.py
from strands import tool
import json
from typing import Optional, Any


def cap_string(json_str, max_length = 7200):
    if len(json_str) > max_length:
        suffix = '... (truncated)'
        return json_str[:max_length - len(suffix)] + suffix
    return json_str



class AppWorldExecutor:
    """Class to maintain AppWorld state and provide consistent tool functions."""
    
    def __init__(self):
        self.world: Optional[Any] = None
        self._execution_count = 0
        
    def set_world(self, world):
        """Set the current AppWorld instance."""
        self.world = world
        self._execution_count = 0
        
    def get_execute_tool(self):
        """Return an execute tool function that uses the current world instance."""
        
        @tool
        def execute(code: str) -> str:
            """Execute Python code in the AppWorld environment.
            
            Args:
                code: Python code to execute in AppWorld
                
            Returns:
                JSON string with execution output and task completion status
            """
            if self.world is None:
                return json.dumps({
                    "error": "No AppWorld instance set",
                    "success": False
                })
            
            # self._execution_count += 1
            
            try:
                output = self.world.execute(code)
                return cap_string(json.dumps({
                    "output": output,
                    "task_completed": self.world.task_completed(),
                    "execution_count": self._execution_count,
                    "success": True
                }))
            except Exception as e:
                import traceback
                traceback.print_exc()
                return cap_string(json.dumps({
                    "error_type": type(e).__name__,
                    "execution_count": self._execution_count,
                    "success": False,
                    "error": str(e),
                }))
            
        
        return execute
