from semantic_kernel.functions import kernel_function
import pandas as pd
import os
import glob
from src.core.kernel import build_kernel
from src.utils.prompt_manager import PromptManager
from semantic_kernel.functions import KernelFunctionFromPrompt

class DataAnalystPlugin:
    def __init__(self, kernel):
        self.kernel = kernel
        self.dfs = {}
        self._load_csvs()

    def _load_csvs(self):
        """Loads all CSVs from data/inputs into DataFrames."""
        csv_path = os.path.join(os.getcwd(), "data/inputs/*.csv")
        files = glob.glob(csv_path)
        for f in files:
            basename = os.path.basename(f)
            try:
                self.dfs[basename] = pd.read_csv(f)
                print(f"DataAnalyst: Loaded {basename} ({len(self.dfs[basename])} rows)")
            except Exception as e:
                print(f"DataAnalyst: Failed to load {basename}: {e}")

    @kernel_function(name="AnalyzeData")
    async def analyze(self, query: str) -> str:
        """
        Analyzes quantitative data in CSV files. Use this tool for math, calculations, counting, averages, or retrieving structured data records.
        """
        if not self.dfs:
            return "No CSV files are available for analysis."

        # 1. Prepare Context (Schemas)
        context_str = ""
        for name, df in self.dfs.items():
            context_str += f"\nFile: {name}\nColumns: {list(df.columns)}\nMsg: {df.head(3).to_string()}\n"

        # 2. Generate Code using LLM
        prompt_template = PromptManager.get("csv_analyst")
        
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

        # Configure execution settings for the 'tools' service
        exec_settings = OpenAIChatPromptExecutionSettings(service_id="tools")

        func = KernelFunctionFromPrompt(
            function_name="generate_code",
            plugin_name="DataAnalyst",
            prompt=prompt_template,
            execution_settings=exec_settings
        )
        
        code_res = await self.kernel.invoke(func, user_input=query, data_context=context_str)
        code = str(code_res).strip().replace("```python", "").replace("```", "").strip()
        
        print(f"Generated Analysis Code:\n{code}")

        # 3. Execute Code
        try:
            local_vars = {"dfs": self.dfs, "pd": pd}
            exec(code, {}, local_vars)
            
            result = local_vars.get("result", "No result variable found.")
            return str(result)
        except Exception as e:
            return f"Analysis Failed: {e}"
