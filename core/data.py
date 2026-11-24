"""Sample paragraph, gold paragraphs and the 50-sentence, 5-topic test set."""

SAMPLE_PARAGRAPH = (
    "The central bank raised interest rates by half a percentage point on Wednesday, its third increase this year, "
    "as policymakers tried to cool inflation that has stayed well above target. Mortgage costs climbed almost "
    "immediately, and analysts warned that the housing market could slow sharply as borrowing becomes more "
    "expensive. Stock markets fell on the news, with bank shares the only sector to gain. The governor said the "
    "labour market remained tight and wage growth was still too strong, but hinted that the pace of tightening "
    "may ease if consumer prices continue to stabilise. Economists are divided on whether the economy can avoid "
    "a recession next year."
)
SAMPLE_GOLD = ["interest rates", "central bank", "inflation", "mortgage costs", "housing market",
               "recession", "monetary tightening", "wage growth", "stock markets", "consumer prices"]

TOPICS = ["Finance", "Sports", "Health", "Technology", "Cooking"]

SENTENCES: list[tuple[str, str]] = [
    ("Finance", "The central bank raised interest rates to fight inflation."),
    ("Finance", "Bond yields climbed after the jobs report beat expectations."),
    ("Finance", "Investors moved money into gold as markets became volatile."),
    ("Finance", "The company's quarterly earnings exceeded analyst forecasts."),
    ("Finance", "Mortgage rates have reached their highest level in a decade."),
    ("Finance", "A weaker currency makes imported goods more expensive."),
    ("Finance", "The stock index fell three percent in a single trading session."),
    ("Finance", "Diversifying a portfolio reduces exposure to any single asset."),
    ("Finance", "Venture capital funding for startups dropped sharply this year."),
    ("Finance", "The government issued new bonds to cover the budget deficit."),
    ("Sports", "The striker scored twice in the second half to win the match."),
    ("Sports", "She set a new world record in the 400 metre freestyle."),
    ("Sports", "The coach was sacked after a run of six straight defeats."),
    ("Sports", "Fans packed the stadium for the championship final."),
    ("Sports", "The tennis star retired with a knee injury in the third set."),
    ("Sports", "The batting side collapsed and lost eight wickets before lunch."),
    ("Sports", "A late three-pointer sent the basketball game into overtime."),
    ("Sports", "The cyclist took the yellow jersey after the mountain stage."),
    ("Sports", "The club signed a young midfielder on a five-year contract."),
    ("Sports", "The marathon runner collapsed metres from the finish line."),
    ("Health", "Regular exercise lowers the risk of heart disease."),
    ("Health", "The new vaccine reduced hospital admissions by half."),
    ("Health", "Doctors recommend at least seven hours of sleep each night."),
    ("Health", "High blood pressure often has no obvious symptoms."),
    ("Health", "The clinical trial tested a drug for early Alzheimer's disease."),
    ("Health", "Eating more fibre can improve gut health."),
    ("Health", "Antibiotic resistance is a growing threat to public health."),
    ("Health", "The patient recovered quickly after knee replacement surgery."),
    ("Health", "Chronic stress can weaken the immune system."),
    ("Health", "Smoking is the leading cause of preventable lung cancer."),
    ("Technology", "The new smartphone ships with a faster processor and better camera."),
    ("Technology", "Engineers trained a neural network to translate speech in real time."),
    ("Technology", "A software update fixed the security vulnerability."),
    ("Technology", "Cloud computing lets companies rent servers by the hour."),
    ("Technology", "The startup built a chip designed for machine learning workloads."),
    ("Technology", "Hackers stole millions of passwords in the data breach."),
    ("Technology", "Electric cars rely on large lithium-ion battery packs."),
    ("Technology", "The open-source library was downloaded a million times."),
    ("Technology", "Quantum computers could break today's encryption schemes."),
    ("Technology", "The app crashed when too many users logged in at once."),
    ("Cooking", "Sear the steak in a hot pan before finishing it in the oven."),
    ("Cooking", "Knead the dough for ten minutes until it is smooth and elastic."),
    ("Cooking", "Add a pinch of salt to bring out the sweetness of the tomatoes."),
    ("Cooking", "Simmer the stock gently for three hours to extract flavour."),
    ("Cooking", "Whisk the egg whites until they form stiff peaks."),
    ("Cooking", "Toast the spices in a dry pan before grinding them."),
    ("Cooking", "Let the bread cool completely before slicing it."),
    ("Cooking", "Marinate the chicken overnight in yoghurt and garlic."),
    ("Cooking", "Caramelise the onions slowly over low heat."),
    ("Cooking", "Fold the chocolate into the cream without knocking out the air."),
]

GOLD_PARAGRAPHS = [
    {"text": SAMPLE_PARAGRAPH, "gold": SAMPLE_GOLD},
    {
        "text": (
            "Researchers have developed a new battery chemistry that could double the range of electric vehicles. "
            "The solid-state design replaces the flammable liquid electrolyte with a ceramic layer, making the "
            "cells safer and allowing faster charging. Car makers are racing to commercialise the technology, "
            "though manufacturing costs remain high and production at scale is still several years away. "
            "Analysts say cheaper, longer-lasting batteries are essential if electric cars are to replace petrol "
            "models."
        ),
        "gold": ["solid-state battery", "electric vehicles", "battery chemistry", "electrolyte", "fast charging",
                 "car makers", "manufacturing costs", "driving range", "petrol cars", "safety"],
    },
    {
        "text": (
            "A large study of more than 100,000 adults found that people who walked at least 7,000 steps a day had "
            "a significantly lower risk of early death. The benefits levelled off above about 10,000 steps, "
            "suggesting that moderate activity is enough for most of the health gains. Walking speed mattered less "
            "than total distance. The researchers say the findings should encourage sedentary people that small "
            "increases in daily movement can improve heart health and longevity."
        ),
        "gold": ["daily steps", "walking", "early death", "physical activity", "heart health", "longevity",
                 "sedentary lifestyle", "health study", "mortality risk", "exercise"],
    },
]
