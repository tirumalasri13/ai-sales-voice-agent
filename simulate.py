"""Scripted end-to-end demo (no browser/mic needed):  python simulate.py
Proves the full workflow: greet -> qualify -> RAG answer -> objection -> book -> CRM -> summary -> insights."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crm  # noqa: E402

crm.DB = os.path.join(HERE, "demo_crm.db")
if os.path.exists(crm.DB):
    os.remove(crm.DB)
crm.init()
import agent  # noqa: E402


def run(lines, lang="en"):
    c = agent.start_call(lang)
    print(f"\n=== CALL {c['call_id']} ===\nAGENT: {c['reply']}")
    for u in lines:
        r = agent.handle_turn(c["call_id"], u)
        print(f"USER : {u}\nAGENT: {r['reply']}\n       [{r['intent']} | sent {r['sentiment']['label']} | "
              f"lead {r['lead']['tier']} {r['lead']['score']} | lang {r['lang']} | rag {[s['title'] for s in r['sources']]}]")
        if r["end_call"]:
            break
    s = agent.end_call(c["call_id"])
    print("SUMMARY:", s["summary"])
    return s


if __name__ == "__main__":
    run(["Hi, I'm interested in solar for my house. My bill is around Rs 4500 a month",
         "I live in Ludhiana. How much does a 3 kW system cost?",
         "That sounds expensive honestly",
         "ok sure, book a site visit",
         "My name is Rahul",
         "9876543210",
         "tomorrow at 11 am"])
    run(["నమస్కారం, నాకు సోలార్ కావాలి. 3 కిలోవాట్ ధర ఎంత?",
         "నా ఇల్లు హైదరాబాద్ లో ఉంది, కరెంటు బిల్లు 4000 రూపాయలు",
         "బుక్ చేయండి",
         "నా పేరు రవి",
         "9123456780",
         "రేపు మధ్యాహ్నం 2 గంటలకు"], lang="te")
    run(["Hello, mujhe solar chahiye. kitna kharcha aayega?",
         "mera naam Simran hai",
         "haan book kar do kal 11 baje",
         "9812345678"], lang="hi")
    run(["Not interested, please don't call again"])
    print("\nINSIGHTS:", json.dumps(agent.insights(), indent=2))
    print("BOOKINGS:", crm.rows("SELECT slot,status FROM bookings"))
