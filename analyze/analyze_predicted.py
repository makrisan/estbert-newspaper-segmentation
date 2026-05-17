"""
Skript segmenteeritud väljundfailide analüüsimiseks. Arvutab pikkuste jaotuse ja top-10 märksõnad.
Kasutamine:
    python analyze_predicted.py \
        --paevaleht  data/output/paevaleht_199109-10_lk_predicted.txt \
        --kommunist  data/output/parnukommunist_195809-10_lk_predicted.txt

Väljundid:
    1. Artiklite pikkuste jaotus (ridade arv artikli kohta)
    2. Kümme sagedamat sisukat märksõna mõlemas ajalehes
"""

import re
import argparse
from collections import Counter


# ------------------------------------------------------------------
# Stoppsõnad — laiendage vajadusel
# ------------------------------------------------------------------
STOPWORDS = {
    "ning", "kuid", "sest", "oma", "mis", "mida", "pole", "veel", "eile", "täna",
    "nagu", "seda", "selle", "neid", "kõik", "ühes", "tema", "nende", "mille",
    "need", "alla", "üles", "ette", "koha", "peab", "teatas", "teatab", "vastu",
    "pärast", "olnud", "olema", "saab", "sõnul", "teatel", "samuti", "selleks",
    "käesoleval", "aastal", "aastas", "kollektiiv", "kohta", "kuna", "jaoks",
    "meie", "seal", "kelle", "nüüd", "välja", "võtab", "saama", "teiste",
    "ütles", "küll", "juba", "mitte", "vaid", "ilma", "läbi", "alles", "peale",
    "juurde", "ainult", "otse", "keegi", "sinna", "sealt", "enne", "uuest",
    "kogu", "igast", "tegu", "sama", "osas", "neis", "teine", "esimest",
    "esimese", "tähele", "korra", "algul", "algus", "sest", "koos", "ilma",
    "mõned", "mõne", "mõni", "võib", "tuleb", "said", "sain", "olid", "oleks",
    "öeldi", "öeldes", "lisas", "märkis", "rääkis", "ütleb", "kirjutab",
    "aasta", "aastaks", "aastatel", "aastani", "kuud", "kuul", "päeval",
    "päeva", "päevil", "nädal", "nädalal", "kell", "seni", "seega",
    "sellel", "sellist", "sellise", "sellega", "selles", "sellest",
    "siia", "siit", "sealt", "sealse", "kõige", "rohkem", "vähem",
    "palju", "väga", "eriti", "möödunud", "järgmisel", "viimane", "viimased",
    "esimene", "kolmas", "uue", "uued", "uus", "vana", "vanad",
    "eesli", "rootsi", "võitis", "jooksul", "enam", "tuli", "siis", "eest",
    "praegu", "ajal", "siiski", "kuigi", "ilmselt", "ühtlasi", "vahel",
    "uudised", "eestis", "siin", "poolt", "tänavu",
    "nende", "enda", "ennast", "meid", "teile", "neile", "meile", "talle", "temale", "sellele",
    "siis", "eest", "praegu", "ajal", "korrespondent", "järgi", "juures",
    "millel", "millega", "millest", "milles", "olles", "oleval", "oleva",
    "peaks", "peaks", "tohib", "tahab", "saaks", "teeb", "teha", "tehtud",
    "antud", "võetud", "pandud", "toodud", "viidud", "jäänud", "jättes",
    "tulles", "minnes", "tulnud", "läinud", "hakanud", "pidanud",
}


def parse_articles(filepath: str) -> list[str]:
    """Loeb predicted.txt ja tagastab artiklite tekstide loendi."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    articles = re.findall(r"<p>\n(.*?)\n</p>", content, re.DOTALL)
    return [a.strip() for a in articles]


def length_distribution(articles: list[str]) -> dict:
    """Arvutab artiklite pikkuste jaotuse ridade arvu järgi."""
    lengths = [len(a.split("\n")) for a in articles]
    total = len(lengths)
    buckets = {
        "1 rida":    sum(1 for l in lengths if l == 1),
        "2-5 rida":  sum(1 for l in lengths if 2 <= l <= 5),
        "6-15 rida": sum(1 for l in lengths if 6 <= l <= 15),
        "16-30 rida":sum(1 for l in lengths if 16 <= l <= 30),
        "30+ rida":  sum(1 for l in lengths if l > 30),
    }
    avg = sum(lengths) / total if total else 0
    return {
        "total_articles": total,
        "avg_lines": round(avg, 1),
        "min_lines": min(lengths),
        "max_lines": max(lengths),
        "buckets": {k: (v, round(100 * v / total, 1)) for k, v in buckets.items()},
    }


def top_keywords(articles: list[str], n: int = 10) -> list[tuple]:
    """Leiab n sagedamat sisukat märksõna (min 4 tähemärki, ei ole stoppsõna)."""
    all_text = " ".join(articles).lower()
    words = re.findall(r"\b[a-züõöä]{4,}\b", all_text)
    freq = Counter(w for w in words if w not in STOPWORDS)
    return freq.most_common(n)


def print_results(name: str, articles: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({len(articles)} artiklit)")
    print(f"{'='*60}")

    dist = length_distribution(articles)
    print(f"\nArtiklite pikkuste jaotus:")
    print(f"  Kokku artikleid : {dist['total_articles']}")
    print(f"  Keskmine pikkus : {dist['avg_lines']} rida")
    print(f"  Min / Max       : {dist['min_lines']} / {dist['max_lines']} rida")
    print(f"\n  {'Vahemik':<12} {'Arv':>6}  {'%':>6}")
    print(f"  {'-'*28}")
    for bucket, (count, pct) in dist["buckets"].items():
        print(f"  {bucket:<12} {count:>6}  {pct:>5.1f}%")

    print(f"\nTop 10 märksõna:")
    print(f"  {'Järk':<5} {'Märksõna':<20} {'Sagedus':>8}")
    print(f"  {'-'*36}")
    for i, (word, count) in enumerate(top_keywords(articles), 1):
        print(f"  {i:<5} {word:<20} {count:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Analüüsib predicted.txt faile: pikkuste jaotus ja märksõnad"
    )
    parser.add_argument(
        "--paevaleht",
        required=True,
        help="Päevalehe predicted.txt faili tee",
    )
    parser.add_argument(
        "--kommunist",
        required=True,
        help="Pärnu Kommunisti predicted.txt faili tee",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Mitu top-märksõna näidata (vaikimisi 10)",
    )
    args = parser.parse_args()

    pd_articles = parse_articles(args.paevaleht)
    pk_articles = parse_articles(args.kommunist)

    print_results("Päevaleht 1991 (september–oktoober)", pd_articles)
    print_results("Pärnu Kommunist 1958 (september–oktoober)", pk_articles)

    print(f"\n{'='*60}")
    print("Kõrvutav kokkuvõte")
    print(f"{'='*60}")
    print(f"\n  {'Näitaja':<35} {'Päevaleht':>12} {'PK 1958':>12}")
    print(f"  {'-'*60}")
    pd_d = length_distribution(pd_articles)
    pk_d = length_distribution(pk_articles)
    print(f"  {'Artikleid kokku':<35} {pd_d['total_articles']:>12} {pk_d['total_articles']:>12}")
    print(f"  {'Keskmine pikkus (rida)':<35} {pd_d['avg_lines']:>12} {pk_d['avg_lines']:>12}")
    for bucket in pd_d["buckets"]:
        pd_v, pd_p = pd_d["buckets"][bucket]
        pk_v, pk_p = pk_d["buckets"][bucket]
        print(f"  {bucket:<35} {pd_v:>6} ({pd_p:>4.1f}%) {pk_v:>6} ({pk_p:>4.1f}%)")


if __name__ == "__main__":
    main()
