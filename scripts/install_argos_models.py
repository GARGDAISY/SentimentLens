"""Install offline Argos Translate models used by Review Sentiment Analyzer.

Default: Mandarin Chinese, Hindi, Spanish, Arabic, French, Bengali -> English.
"""
from __future__ import annotations

import argparse
import sys

try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    raise SystemExit("Argos Translate is missing. Run: python -m pip install argostranslate")

DEFAULT_LANGUAGES = {
    "zh": "Mandarin Chinese",
    "hi": "Hindi",
    "es": "Spanish",
    "ar": "Arabic",
    "fr": "French",
    "bn": "Bengali",
}


def installed_pairs() -> set[tuple[str, str]]:
    pairs = set()
    for language in argostranslate.translate.get_installed_languages():
        for translation in getattr(language, "translations", []):
            pairs.add((language.code, translation.to_lang.code))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Install offline Argos language-to-English models.")
    parser.add_argument("--without-hindi", action="store_true", help="Skip Hindi -> English (not recommended for full 7-language support).")
    args = parser.parse_args()
    requested = dict(DEFAULT_LANGUAGES)
    if args.without_hindi:
        requested.pop("hi", None)

    print("Updating Argos package index (internet required for this setup step)...")
    try:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
    except Exception as exc:
        print(f"Could not retrieve Argos package index: {exc}", file=sys.stderr)
        return 1

    installed = installed_pairs()
    failed = []
    for code, name in requested.items():
        if (code, "en") in installed:
            print(f"[already installed] {name} -> English")
            continue
        package = next((p for p in available if p.from_code == code and p.to_code == "en"), None)
        if package is None:
            print(f"[not available] {name} -> English")
            failed.append(name)
            continue
        try:
            print(f"[downloading] {name} -> English")
            path = package.download()
            argostranslate.package.install_from_path(path)
            print(f"[installed] {name} -> English")
        except Exception as exc:
            print(f"[failed] {name}: {exc}", file=sys.stderr)
            failed.append(name)

    print("\nSetup finished. Translation models are stored in Argos' local model directory.")
    if failed:
        print("Could not install: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
