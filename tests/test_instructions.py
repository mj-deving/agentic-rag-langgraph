import hashlib

from agentic_rag_langgraph.instructions import INSTRUCTIONS

# SHA-256 of INSTRUCTIONS.v1 / .v2 rendered by the TypeScript original
# (src/mastra/agents/instructions.ts in the Mastra repo, at the commit in corpus/PROVENANCE.json).
TS_HASHES = {
    "v1": "1d772234d9244c2f15cc936a869a6eea7d774d67829430411749ff603e7028b6",
    "v2": "16739a8618167b8b2ac87d74efed8817be0a0e6f01d47a9fe1d20237a3b766ef",
}


def test_prompts_are_byte_identical_to_the_typescript_original():
    for version, digest in TS_HASHES.items():
        assert hashlib.sha256(INSTRUCTIONS[version].encode()).hexdigest() == digest, version
