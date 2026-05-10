from src.text_cleaner import protect_abbreviations, restore_abbreviations, split_into_sentences

def show(label: str, original: str):
    protected = protect_abbreviations(original)
    split     = split_into_sentences([original])
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  Algne:    {repr(original)}")
    print(f"  Kaitstud: {repr(protected)}")
    print(f"  Laused ({len(split)}):")
    for s in split:
        print(f"    → {s}")

# Kuupäevad üle rea (sinu näidetest)
show("Aasta üle rea",
     "võrdleme 2007.\nja 2011. aasta Riigikogu valimisi.")

show("Kuupäev + kuu lühend üle rea",
     "Sündinud 26.\nveebr.\n1964 2011- ...")

show("Kuupäev + täiskuu üle rea",
     "23.\noktoobril 1989 marssis 300 000 demonstranti.")

# Loendi numbrid üle rea
show("Loendi number üle rea",
     "9.\nnovembril varises Berliini müür.")

show("Raamatu number",
     "1.\nraamat: Mälestusi kodumaalt")

# Isiku initsiaalid
show("Initsiaal + perekonnanimi",
     "laulis C.\nKreek, R.\nValgre, U.\nNaissoo laule.")

show("N. Liidu initsiaal",
     "kirjeldab **N.\nLiidu** riiklikult dirigeeritud majandust.")

# Lühendid
show("nn. lühend",
     "olid ainult **nn.**\nrahvaesinduse valimised.")

show("hilj. pühap. tel.",
     "teatada perenaisele hilj.\npühap.\ntel.\n046211 80 04")

# Nädalapäevad layout_noise filtris
from src.text_cleaner import is_layout_noise
for day in ["Esmaspäev", "Teisipäev", "Kolmapäev", "Neljapäev", "Reede", "Laupäev", "Pühapäev"]:
    line = f"{day}, 12. mai 2011"
    result = is_layout_noise(line)
    status = "✅ noise" if result else "❌ EI tuvastatud"
    print(f"\nNädalapäev test: '{line}' → {status}")

print("\n✅ Testid lõpetatud.")
