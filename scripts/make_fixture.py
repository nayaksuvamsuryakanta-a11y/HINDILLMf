from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hindi_llm import make_fixture
make_fixture()
print("Created tests/fixtures/sample_hi.txt")
