"""Minimal example: store knowledge and ask a question."""

from offline_ai import LocalAI

ai = LocalAI("./workspace")

ai.add("Company X announced that its VPN infrastructure was compromised.")
ai.add(
    [
        "Threat actors used CVE-2024-12345 against Company X.",
        "Phishing campaign targets banks in Germany.",
    ]
)

print(ai.ask("Find reports about Company X attacks", text_only=True))

result = ai.ask("Find reports about Company X attacks")
print(result["answer"])
print(result["documents"])
