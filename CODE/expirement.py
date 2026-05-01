!pip install google-genai==1.66.0 -q

from google import genai
import pandas as pd
import matplotlib.pyplot as plt
import random, re, time
from datasets import load_dataset

API_KEY = "AIzaSyBeDcY3ggvchvlV4oaTiCC_ViqsJxMmsjs"

client = genai.Client(api_key=API_KEY)

random.seed(42)


print("setup done")

noise = [
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    "Napoleon Bonaparte was exiled to Saint Helena in 1815.",
    "The Amazon River discharges more water than any other river on Earth.",
    "Ludwig van Beethoven composed his Ninth Symphony while completely deaf.",
    "The speed of light in a vacuum is approximately 299,792 km per second.",
]

def ask(prompt, retries=5):
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model="models/gemma-3-1b-it",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print("API ERROR:", e)
            wait_time = 8 * (i + 1)
            print(f"Retrying in {wait_time} sec...")
            time.sleep(wait_time)
    return ""

results = []

for i, ex in enumerate(gsm):
    q       = ex["question"]
    answer  = ex["answer"].split("####")[-1].strip()
    r_noise = random.choice(noise)

    p_min = f"Question: {q}\nAnswer with just the final number:"
    p_rel = f"Solve step by step.\nQuestion: {q}\nAnswer:"
    p_irr = f"Context: {r_noise}\nQuestion: {q}\nAnswer with just the final number:"

    for condition, prompt in [("minimal", p_min), ("relevant", p_rel), ("irrelevant", p_irr)]:
        response = ask(prompt)
        nums = re.findall(r'\d+', response.replace(',', ''))
        correct = nums[-1] == answer.replace(',', '') if nums else False

        results.append({
            "dataset": "GSM8K",
            "condition": condition,
            "question": q,
            "answer": answer,
            "response": response,
            "correct": correct
        })

        time.sleep(5)

    print(f"GSM8K done: {i+1}/{len(gsm)}")


for i, ex in enumerate(csqa):
    q       = ex["question"]
    answer  = ex["answerKey"]
    choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
    r_noise = random.choice(noise)

    p_min = f"Question: {q}\n{choices}\nAnswer with just the letter:"
    p_rel = f"Use commonsense knowledge.\nQuestion: {q}\n{choices}\nAnswer with just the letter:"
    p_irr = f"Context: {r_noise}\nQuestion: {q}\n{choices}\nAnswer with just the letter:"

    for condition, prompt in [("minimal", p_min), ("relevant", p_rel), ("irrelevant", p_irr)]:
        response = ask(prompt)
        correct = response.strip().upper().startswith(answer.upper())

        results.append({
            "dataset": "CommonsenseQA",
            "condition": condition,
            "question": q,
            "answer": answer,
            "response": response,
            "correct": correct
        })

        time.sleep(2)

    print(f"CSQA done: {i+1}/{len(csqa)}")


for i, ex in enumerate(hotpot):
    q       = ex["question"]
    answer  = ex["answer"]
    support = " ".join([s for para in ex["context"]["sentences"] for s in para][:3])
    r_noise = random.choice(noise) + " " + random.choice(noise)

    p_min = f"Question: {q}\nAnswer briefly:"
    p_rel = f"Context: {support}\nQuestion: {q}\nAnswer briefly:"
    p_irr = f"Context: {r_noise}\nQuestion: {q}\nAnswer briefly:"

    for condition, prompt in [("minimal", p_min), ("relevant", p_rel), ("irrelevant", p_irr)]:
        response = ask(prompt)
        correct = answer.lower() in response.lower()

        results.append({
            "dataset": "HotpotQA",
            "condition": condition,
            "question": q,
            "answer": answer,
            "response": response,
            "correct": correct
        })

        time.sleep(5)

    print(f"HotpotQA done: {i+1}/{len(hotpot)}")


df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("DONE")


summary = df.groupby(["dataset", "condition"])["correct"].mean().mul(100).round(1).unstack()
summary = summary[["minimal", "relevant", "irrelevant"]]
print(summary)
summary.to_csv("accuracy_summary.csv")


summary.plot(kind="bar", figsize=(10, 6), color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="white")
plt.title("Accuracy by Context Condition and Task Type\n(Gemini-2.0-Flash, n=50 per dataset)")
plt.xlabel("Dataset")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.legend(["Minimal", "Relevant", "Irrelevant"], title="Condition")
plt.tight_layout()
plt.savefig("figure2_barchart.png", dpi=200)
plt.show()

errors = df[(df["condition"] == "irrelevant") & (df["correct"] == False)]
errors.to_csv("errors_irrelevant.csv", index=False)
correct_minimal = set(df[(df["condition"] == "minimal") & (df["correct"] == True)]["question"])
hurt = df[(df["condition"] == "irrelevant") & (df["correct"] == False) & (df["question"].isin(correct_minimal))]
hurt.to_csv("errors_hurt_by_irrelevant.csv", index=False)
print(f"{len(hurt)} cases where irrelevant hurt")

from google.colab import files
files.download("results.csv")
files.download("accuracy_summary.csv")
files.download("figure2_barchart.png")
files.download("errors_irrelevant.csv")
files.download("errors_hurt_by_irrelevant.csv")