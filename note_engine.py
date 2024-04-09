from llama_index.core.tools import FunctionTool, ToolMetadata
import os

note_file = os.path.join("data", "notes.txt")

metadata = ToolMetadata(
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n"])

    return "note has been saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    #metadata=note_engine_metadata,
    name="note_saver",  
    #description="this tool can save a text based note to a file for the user",
)