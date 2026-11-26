"""HTTP API + static UI.  Run:  python app.py   (or: uvicorn app:app --port 8000)

Voice I/O (STT/TTS) happens in the browser via the Web Speech API for the prototype.
The backend contract is text-in/text-out, so a production telephony/WebRTC pipeline
(Deepgram/Whisper STT -> this service -> ElevenLabs/Azure TTS) plugs into the same endpoints.
"""
import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

import agent
import crm

HERE = os.path.dirname(os.path.abspath(__file__))
crm.DB = os.path.join(HERE, "crm.db")
crm.init()


async def index(request):
    return FileResponse(os.path.join(HERE, "static", "index.html"))


async def start(request):
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    return JSONResponse(agent.start_call(body.get("lang", "en")))


async def turn(request):
    b = await request.json()
    if b.get("call_id") not in agent.SESSIONS:
        return JSONResponse({"error": "unknown call_id"}, status_code=404)
    text = (b.get("text") or "").strip()[:500]
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    return JSONResponse(agent.handle_turn(b["call_id"], text))


async def end(request):
    b = await request.json()
    return JSONResponse(agent.end_call(b.get("call_id", "")))


async def leads(request):
    return JSONResponse(crm.rows("SELECT * FROM leads ORDER BY score DESC"))


async def bookings(request):
    return JSONResponse(crm.rows("""SELECT b.slot, b.status, l.name, l.phone, l.city, l.tier
                                    FROM bookings b LEFT JOIN leads l ON l.id=b.lead_id ORDER BY b.slot"""))


async def calls(request):
    return JSONResponse(crm.rows("SELECT id,language,outcome,tier,score,sentiment_avg,summary,duration FROM calls ORDER BY created_at DESC LIMIT 20"))


async def insights(request):
    return JSONResponse(agent.insights())


async def kb_search(request):
    q = request.query_params.get("q", "")
    return JSONResponse(agent.KB.search(q, k=3))


async def health(request):
    return JSONResponse({"ok": True, "llm": bool(os.getenv("ANTHROPIC_API_KEY"))})


from starlette.staticfiles import StaticFiles
from starlette.routing import Mount

app = Starlette(routes=[
    Route("/", index),
    Route("/api/call/start", start, methods=["POST"]),
    Route("/api/call/turn", turn, methods=["POST"]),
    Route("/api/call/end", end, methods=["POST"]),
    Route("/api/leads", leads),
    Route("/api/bookings", bookings),
    Route("/api/calls", calls),
    Route("/api/insights", insights),
    Route("/api/kb", kb_search),
    Route("/health", health),
    Mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static"),
])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
