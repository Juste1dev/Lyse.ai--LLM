from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyse_ai.inference import generate_text, load_model_and_tokenizer


SYSTEM_PROMPT = (
    "<|system|>\n"
    "Tu es Lyse AI, un assistant utile, concis, francophone et honnête.\n"
)



def main():
    parser = argparse.ArgumentParser(description="Chatbot TUI pour Lyse AI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    console = Console()
    model, tokenizer, _ = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device=args.device)
    session = PromptSession()
    history = []

    console.print(Panel.fit("Lyse AI TUI\nCommandes: /reset /temp 0.7 /top_p 0.9 /max_new 128 /quit", border_style="cyan"))

    while True:
        try:
            user = session.prompt(HTML("<ansicyan><b>Vous ></b></ansicyan> "))
        except (EOFError, KeyboardInterrupt):
            break

        if not user.strip():
            continue
        if user.startswith("/quit"):
            break
        if user.startswith("/reset"):
            history.clear()
            console.print("[yellow]Contexte vidé.[/yellow]")
            continue
        if user.startswith("/temp "):
            args.temperature = float(user.split(maxsplit=1)[1])
            console.print(f"[green]temperature = {args.temperature}[/green]")
            continue
        if user.startswith("/top_p "):
            args.top_p = float(user.split(maxsplit=1)[1])
            console.print(f"[green]top_p = {args.top_p}[/green]")
            continue
        if user.startswith("/max_new "):
            args.max_new_tokens = int(user.split(maxsplit=1)[1])
            console.print(f"[green]max_new_tokens = {args.max_new_tokens}[/green]")
            continue

        history.append({"role": "user", "content": user})
        prompt = SYSTEM_PROMPT
        for turn in history:
            role_token = "<|user|>" if turn["role"] == "user" else "<|assistant|>"
            prompt += f"{role_token}\n{turn['content']}\n"
        prompt += "<|assistant|>\n"

        answer = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
        if "<|assistant|>" in answer:
            answer = answer.split("<|assistant|>")[-1].strip()
        history.append({"role": "assistant", "content": answer})
        console.print(Panel(answer, title="Lyse AI", border_style="green"))


if __name__ == "__main__":
    main()
