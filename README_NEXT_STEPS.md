# Suggested local usage in WSL

## 1. Enter the project folder
```bash
cd ~/path/to/block-prefix-analyzer
```

## 2. Save the provided files into the repo root
Copy these files into the repository root:
- `PROJECT_SPEC_FOR_CLAUDE.md`
- `FIRST_PROMPT_FOR_CLAUDE.md`

## 3. Confirm your local environment
Recommended checks:
```bash
pwd
python3 --version
git --version
pytest --version
```

If `pytest` is missing:
```bash
python3 -m pip install pytest
```

## 4. Start Claude Code in this folder
```bash
claude
```

## 5. In Claude Code, first ask it to read the spec and only scaffold the repo
Paste the contents of `FIRST_PROMPT_FOR_CLAUDE.md`.

## 6. Expected collaboration style
You have already constrained Claude Code to:
- think in English
- reply to you in Chinese
- prioritize stability, extensibility, test isolation, and module separation
- use Git in small, reviewable steps

## 7. Review its output before asking for the next round
Do not let it implement tokenizer / vLLM alignment yet.
Keep the first round limited to:
- spec review
- implementation plan
- git initialization
- repo skeleton
- test scaffolding