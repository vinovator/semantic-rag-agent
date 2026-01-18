from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from src.core.kernel import build_kernel
from src.core.config import load_config
from src.plugins.rag_plugin import AdvancedRagPlugin
from src.plugins.data_plugin import DataAnalystPlugin
from src.utils.prompt_manager import PromptManager

class RAGAgent:
    def __init__(self):
        self.config = load_config()
        self.kernel, self.embed_service = build_kernel()
        
        # Initialize Plugins
        # Initialize Plugins
        self.rag_plugin = AdvancedRagPlugin(self.embed_service, self.config)
        self.data_plugin = DataAnalystPlugin(self.kernel)
        
        self.kernel.add_plugin(self.rag_plugin, plugin_name="AdvancedRagPlugin")
        self.kernel.add_plugin(self.data_plugin, plugin_name="DataAnalystPlugin")
        
        PromptManager.load()
        
        # Initialize History & System Prompt (Once)
        self.history = ChatHistory()
        self._setup_system_prompt()

    def _setup_system_prompt(self):
        # Grounding: Get list of available files
        import os
        import glob
        
        # CSVs (from Plugin)
        csv_files = ", ".join(self.data_plugin.dfs.keys()) if hasattr(self, 'data_plugin') and self.data_plugin.dfs else "None"
        
        # PDFs (from Disk)
        pdf_path = os.path.join(os.getcwd(), "data/inputs/*.pdf")
        pdf_files = ", ".join([os.path.basename(f) for f in glob.glob(pdf_path)])
        
        # Inject into Prompt
        system_prompt_template = PromptManager.get("tool_chat_system")
        system_prompt = system_prompt_template.format(csv_files=csv_files, pdf_files=pdf_files)
        
        self.history.add_system_message(system_prompt)

    async def process_query(self, user_input: str) -> dict:
        """
        Uses Automatic Function Calling to handle the query.
        """
        try:
            # 1. Update History
            self.history.add_user_message(user_input)

            # 2. Configure Tools (Auto-Invoke)
            settings = PromptExecutionSettings(
                service_id="agent",
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )

            # 3. Get Chat Service
            chat_service = self.kernel.get_service("agent")
            
            # 4. Invoke LLM
            result = await chat_service.get_chat_message_content(
                chat_history=self.history,
                settings=settings,
                kernel=self.kernel
            )
            
            print(f"DEBUG: LLM Content: '{result.content}'")
            print(f"DEBUG: LLM Metadata: {result.metadata}")
            for item in result.items:
                print(f"DEBUG: LLM Item: {type(item)} -> {item}")
            
            # 5. Commit Assistant Response to History
            self.history.add_message(result)

            return {
                "answer": str(result),
                "thought_process": {
                    "mode": "auto_tool_calling",
                    "final_response": str(result)
                }
            }

        except Exception as e:
            print(f"Agent Error: {e}")
            return {"answer": "An internal error occurred.", "error": str(e)}