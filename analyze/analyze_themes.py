"""
Skript artiklite illustratiivseks temaatiliseks kategoriseerimiseks. Arvutab temaatilise jaotuse.
Kasutamine:
    python analyze_themes.py \
        --paevaleht  data/output/paevaleht_199109-10_lk_predicted.txt \
        --kommunist  data/output/parnukommunist_195809-10_lk_predicted.txt

NB! Tegemist on illustratiivse analüüsiga — kategooriad põhinevad
lihtsal märksõnaloendil, mitte täielikul teemamudeldamisel.
Märksõnaloendeid saab faili lõpus muuta.
"""

import re
import argparse
from collections import defaultdict


# ------------------------------------------------------------------
# Teemakategooriad koos märksõnaloendiga
# Iga artikkel saab esimese kategooria, mille märksõna tekstis leidub.
# Järjestus mõjutab tulemusi — spetsiifilisemad kategooriad pange ette.
# ------------------------------------------------------------------
THEMES = {
    "Välispoliitika / rahvusvaheline": [
        "välisministr", "suursaadik", "diplomaati", "välisriik", "välismaa",
        "välispolii", "rahvusvahe", "euroopa ühendus", "nato", "üro",
        "ühendriigid", "suurbritannia", "prantsusmaa", "saksamaa", "jaapan",
    ],
    "Eesti iseseisvus / sisepoliitika": [
        "iseseisvus", "vabariik", "eesti", "balti", "tunnustami", "suveräänsu",
        "ülemnõukogu", "rahvasaadik", "riigikogu", "valitsus",
    ],
    "Nõukogude Liit / Venemaa": [
        "nõukogude liit", "liidus", "N. Liit", "N. Liidu", "moskva", "venemaa", "liiduvabariik",
        "nsvl", "kommunistlik partei", "gorbatšov", "jeltsini",
    ],
    "Sõda / relvastuskonflikt": [
        "sõda", "tulevahetus", "relvad", "sõjaväe", "föderaalarm",
        "horvaatia", "jugoslaavia", "iraak", "kuveit", "omon",
    ],
    "Majandus / tööstus": [
        "majandus", "tootmine", "tehas", "ettevõte", "kombinaat",
        "vabrik", "toodang", "plaan", "eksport", "import", "kasum",
    ],
    "Põllumajandus / kalandus": [
        "kolhoos", "kalapüük", "räim", "kala", "põllumajan",
        "saak", "põld", "vili", "heeringa", "traaler",
    ],
    "Kultuur / haridus / sport": [
        "kultuur", "kool", "õpil", "haridus", "sport", "rock",
        "kontsert", "laul", "õpetaj", "ülikool", "muuseum",
    ],
    "Sotsialism / tootmisvõistlus": [
        "sotsialisti", "võistlus", "kohustus", "plaani täitmi",
        "suure oktoobri", "kommunist", "brigaad", "normi",
    ],
    "Kohalik elu": [
        "tallinn", "pärnu", "linna", "rakvere", "tartu", "kohalik",
        "vald", "linnavolikogu",
    ],
}


def parse_articles(filepath: str) -> list[str]:
    """Loeb predicted.txt ja tagastab artiklite tekstide loendi."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    articles = re.findall(r"<p>\n(.*?)\n</p>", content, re.DOTALL)
    return [a.strip() for a in articles]


def categorize(articles: list[str]) -> dict:
    """Omistab igale artiklile teema esimese sobiva märksõna järgi."""
    counts = defaultdict(int)
    uncategorized = 0
    for article in articles:
        text = article.lower()
        matched = False
        for theme, keywords in THEMES.items():
            if any(kw in text for kw in keywords):
                counts[theme] += 1
                matched = True
                break
        if not matched:
            uncategorized += 1
    counts["Kategoriseerimata"] = uncategorized
    return dict(counts)


def print_theme_table(name: str, articles: list[str]) -> None:
    total = len(articles)
    counts = categorize(articles)

    print(f"\n{'='*60}")
    print(f"  {name}  (kokku {total} artiklit)")
    print(f"{'='*60}")
    print(f"\n  {'Teema':<40} {'Arv':>5}  {'%':>6}")
    print(f"  {'-'*55}")

    # Sorteeri sageduse järgi, kategoriseerimata viimane
    sorted_items = sorted(
        [(k, v) for k, v in counts.items() if k != "Kategoriseerimata"],
        key=lambda x: -x[1],
    )
    sorted_items.append(("Kategoriseerimata", counts.get("Kategoriseerimata", 0)))

    for theme, count in sorted_items:
        pct = 100 * count / total
        print(f"  {theme:<40} {count:>5}  {pct:>5.1f}%")


def print_comparison(pd_articles, pk_articles):
    pd_total = len(pd_articles)
    pk_total = len(pk_articles)
    pd_counts = categorize(pd_articles)
    pk_counts = categorize(pk_articles)

    all_themes = list(THEMES.keys()) + ["Kategoriseerimata"]

    print(f"\n{'='*70}")
    print("  Kõrvutav temaatiline jaotus")
    print(f"{'='*70}")
    print(f"\n  {'Teema':<40} {'PL 1991':>10} {'PK 1958':>10}")
    print(f"  {'-'*62}")
    for theme in all_themes:
        pd_pct = 100 * pd_counts.get(theme, 0) / pd_total
        pk_pct = 100 * pk_counts.get(theme, 0) / pk_total
        print(f"  {theme:<40} {pd_pct:>9.1f}% {pk_pct:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Illustratiivne temaatiline analüüs predicted.txt failidest"
    )
    parser.add_argument("--paevaleht", required=True)
    parser.add_argument("--kommunist", required=True)
    args = parser.parse_args()

    pd_articles = parse_articles(args.paevaleht)
    pk_articles = parse_articles(args.kommunist)

    print_theme_table("Päevaleht 1991 (september–oktoober)", pd_articles)
    print_theme_table("Pärnu Kommunist 1958 (september–oktoober)", pk_articles)
    print_comparison(pd_articles, pk_articles)

    print(f"\nNB! Tegemist on illustratiivse analüüsiga.")
    print(f"Märksõnaloendeid saab skripti failis THEMES sõnastikus muuta.")


if __name__ == "__main__":
    main()
