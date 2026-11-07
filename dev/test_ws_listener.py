import asyncio
import json
import websockets

async def listen():
    uri = "ws://127.0.0.1:8188/ws?clientId=listener_probe"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as ws:
            print("Connected. Listening for 10 seconds...")
            for _ in range(20):
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                msg_type = data.get("type")
                msg_data = data.get("data")
                if msg_type in ("progress", "executing", "executed", "status", "execution_error"):
                    print(f"[{msg_type}] {msg_data}")
    except asyncio.TimeoutError:
        print("Timeout waiting for message.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(listen())
